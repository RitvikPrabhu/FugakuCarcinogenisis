#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <set>
#include <vector>

#include "multiHit.h"
#include "utils.h"

//////////////////////////////  Start Allreduce_hierarchical
/////////////////////////

#ifdef HIERARCHICAL_COMMS
#include <unistd.h>

#define ALL_REDUCE_FUNC Allreduce_hierarchical
#define EXECUTE execute_hierarchical
static void Allreduce_hierarchical(void *sendbuf, void *recvbuf, int count,
                                   MPI_Datatype datatype, MPI_Op op,
                                   CommsStruct &comms) {

  // Datatype size
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);

  // Allocate buffer for local reduction
  void *local_result = malloc(count * datatype_size);

  // Phase 1: Local Reduction
  MPI_Reduce(sendbuf, local_result, count, datatype, op, 0, comms.local_comm);

  // Allocate buffer for global reduction (only needed for rank 0 in
  // local_comm)
  void *global_result = NULL;
  if (comms.is_leader) {
    global_result = malloc(count * datatype_size);
  }

  // Phase 2: Global Reduction (only rank 0 in each node participates)
  if (comms.is_leader) {
    MPI_Allreduce(local_result, global_result, count, datatype, op,
                  comms.global_comm);
    // Copy the final result to recvbuf
    memcpy(recvbuf, global_result, count * datatype_size);
  }

  // Phase 3: Broadcast result to all processes in the local communicator
  MPI_Bcast(recvbuf, count, datatype, 0, comms.local_comm);

  // Cleanup
  free(local_result);
  if (comms.is_leader)
    free(global_result);
}

static inline WorkChunk calculate_node_range(LAMBDA_TYPE num_Comb,
                                             const CommsStruct &comms) {
  const LAMBDA_TYPE base = num_Comb / comms.num_nodes;
  const LAMBDA_TYPE extra = num_Comb % comms.num_nodes;
  const int k = comms.my_node_id;

  const LAMBDA_TYPE start = k * base + std::min<LAMBDA_TYPE>(k, extra);
  const LAMBDA_TYPE len = base + (k < extra ? 1 : 0);

  return {start, start + len - 1};
}

static inline WorkChunk calculate_worker_range(const WorkChunk &leaderRange,
                                               int worker_id, int num_workers) {

  const LAMBDA_TYPE start = leaderRange.start + worker_id * CHUNK_SIZE;

  if (start > leaderRange.end) {
    return {0, -1};
  }

  const LAMBDA_TYPE end = std::min(start + CHUNK_SIZE - 1, leaderRange.end);

  return {start, end};
}

inline LAMBDA_TYPE length(const WorkChunk &c) { return c.end - c.start + 1; }

inline static void handle_local_work_steal(WorkChunk &availableWork,
                                           MPI_Status st,
                                           const CommsStruct &comms) {

  int requester = st.MPI_SOURCE;
  char dummy;
  MPI_Recv(&dummy, 1, MPI_BYTE, requester, TAG_REQUEST_WORK, comms.local_comm,
           MPI_STATUS_IGNORE);
  WorkChunk reply;
  if (availableWork.start <= availableWork.end) {
    LAMBDA_TYPE chunkEnd =
        std::min(availableWork.start + CHUNK_SIZE - 1, availableWork.end);
    reply = {availableWork.start, chunkEnd};
    availableWork.start += (chunkEnd + 1);

  } else {
    reply = {0, -1};
  }

  MPI_Send(&reply, sizeof(WorkChunk), MPI_BYTE, requester, TAG_ASSIGN_WORK,
           comms.local_comm);
}

inline static void inter_node_work_steal_victim(WorkChunk &availableWork,
                                                MPI_Status st, int &my_color,
                                                Token &tok,
                                                const CommsStruct &comms) {
  printf("BEGIN inter_node_work_steal_victim\n");
  fflush(stdout);
  char dummy;
  MPI_Request rq_recv;
  MPI_Irecv(&dummy, 1, MPI_BYTE, st.MPI_SOURCE, TAG_NODE_STEAL_REQ,
            comms.global_comm, &rq_recv);

  WorkChunk reply;

  if (availableWork.start < availableWork.end) {
    LAMBDA_TYPE len = availableWork.end - availableWork.start + 1;
    LAMBDA_TYPE mid = availableWork.start + len / 2;
    reply = {mid, availableWork.end};
    availableWork.end = mid - 1;
    my_color = BLACK;
  } else {
    reply = {0, -1};
  }

  MPI_Request rq;
  MPI_Isend(&reply, sizeof(WorkChunk), MPI_BYTE, st.MPI_SOURCE,
            TAG_NODE_STEAL_REPLY, comms.global_comm, &rq);
}

static inline void root_broadcast_termination(const CommsStruct &comms,
                                              MPI_Win &term_win) {
  bool termination_signal = true;
  for (int rank = 0; rank < comms.num_nodes; ++rank) {
    MPI_Put(&termination_signal, 1, MPI_C_BOOL, rank, 0, 1, MPI_C_BOOL,
            term_win);
  }
  MPI_Win_flush_all(term_win);
}

static inline void try_forward_token_if_idle(WorkChunk &availableWork,
                                             bool &have_token, int &my_color,
                                             Token &tok, const int next_leader,
                                             MPI_Win &term_win,
                                             const CommsStruct &comms) {

  if (!have_token || length(availableWork) > 0) {
    return;
  }

  if (my_color == BLACK)
    tok.colour = BLACK;

  if (comms.global_rank == 0) {
    if (tok.colour == WHITE && tok.finalRound) {
      root_broadcast_termination(comms, term_win);
    } else {
      tok.finalRound = (tok.colour == WHITE);
      tok.colour = WHITE;
      MPI_Send(&tok, sizeof(Token), MPI_BYTE, next_leader, TAG_TOKEN,
               comms.global_comm);
    }
  } else {
    MPI_Send(&tok, sizeof(Token), MPI_BYTE, next_leader, TAG_TOKEN,
             comms.global_comm);
  }

  have_token = false;
  my_color = WHITE;
}

inline static void receive_token(Token &tok, MPI_Status st, bool &have_token,
                                 const CommsStruct &comms) {

  MPI_Status status;
  MPI_Recv(&tok, sizeof(Token), MPI_BYTE, st.MPI_SOURCE, TAG_TOKEN,
           comms.global_comm, &status);
  have_token = true;
  /**
    if (comms.global_rank == 0) {
      if (tok.colour == WHITE) {
        tok.finalRound = true;
      } else {
        tok.colour = WHITE;
      }
    }**/
}

inline static WorkChunk
assign_and_update_availableWork(const WorkChunk &loot, int num_workers,
                                const CommsStruct &comms) {
  LAMBDA_TYPE max_end = loot.start - 1;
  for (int w = 1; w <= num_workers; ++w) {
    WorkChunk work = calculate_worker_range(loot, w - 1, num_workers);
    printf("[DEBUG assign and update availableWork] Rank %d (Node %d, Local "
           "Rank %d): Assigned chunk [%lld, "
           "%lld]\n",
           comms.global_rank, comms.my_node_id, comms.local_rank, work.start,
           work.end);
    fflush(stdout);

    MPI_Send(&work, sizeof(WorkChunk), MPI_BYTE, w, TAG_ASSIGN_WORK,
             comms.local_comm);
    if (work.end > max_end)
      max_end = work.end;
  }
  if (max_end >= loot.end) {
    return {0, -1};
  } else {
    return {max_end + 1, loot.end};
  }
}

inline static void
inter_node_work_steal_initiate(WorkChunk &availableWork, MPI_Status st,
                               int num_workers, int &my_color, Token &tok,
                               MPI_Win &term_win, bool *global_done,
                               const CommsStruct &comms) {
  printf("BEGIN: inter_node_work_steal_initiate\n");
  fflush(stdout);
  int myRank = comms.global_rank;
  int nLeaders = comms.num_nodes;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> gen(0, nLeaders - 1);
  bool lootReceived = false;
  char dummy;

  for (int attempt = 0; attempt < NUM_RETRIES && !lootReceived; ++attempt) {
    int victim;
    do {
      victim = gen(rng);
    } while (victim == myRank);

    MPI_Request rq;
    MPI_Status status;

    MPI_Isend(&dummy, 1, MPI_BYTE, victim, TAG_NODE_STEAL_REQ,
              comms.global_comm, &rq);

    WorkChunk loot;
    MPI_Request rq_recv;
    MPI_Irecv(&loot, sizeof(WorkChunk), MPI_BYTE, victim, TAG_NODE_STEAL_REPLY,
              comms.global_comm, &rq_recv);

    int completed = 0;
    MPI_Win_sync(term_win);

    while (!completed && !(*global_done)) {
      MPI_Test(&rq_recv, &completed, MPI_STATUS_IGNORE);
      int flag = 0;
      MPI_Status st;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms.global_comm, &flag, &st);

      if (flag) {
        switch (st.MPI_TAG) {

        case TAG_NODE_STEAL_REQ:
          MPI_Wait(&rq, &status);
          inter_node_work_steal_victim(availableWork, st, my_color, tok, comms);
          break;
        }
      }
      MPI_Win_sync(term_win);
    }
    MPI_Wait(&rq, &status);

    if (length(loot) > 0) {
      availableWork = assign_and_update_availableWork(loot, num_workers, comms);
      lootReceived = true;
    }
  }
}

static void send_poison_pill(int num_workers, const CommsStruct &comms) {
  WorkChunk poison{0, -2};
  for (int w = 1; w <= num_workers; ++w) {
    MPI_Send(&poison, sizeof(poison), MPI_BYTE, w, TAG_ASSIGN_WORK,
             comms.local_comm);
  }
}

static void node_leader_hierarchical(WorkChunk availableWork, int num_workers,
                                     const CommsStruct &comms) {

  bool *global_done;
  MPI_Win term_win;
  MPI_Win_allocate(sizeof(bool), sizeof(bool), MPI_INFO_NULL, comms.global_comm,
                   &global_done, &term_win);
  *global_done = false;
  MPI_Win_lock_all(0, term_win);

  const int next_leader = (comms.global_rank + 1) % comms.num_nodes;
  Token tok = {WHITE, false};
  bool have_token = (comms.global_rank == 0);
  int my_color = WHITE;

  while (true) {
    MPI_Win_sync(term_win);
    if (*global_done) {
      break;
    }
    MPI_Status st;
    int flag;
    // local probe
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms.local_comm, &flag, &st);
    if (flag) {
      int tag = st.MPI_TAG;
      switch (tag) {
      case TAG_REQUEST_WORK:
        handle_local_work_steal(availableWork, st, comms);
        break;
      }
    }

    // global probe
    flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms.global_comm, &flag, &st);
    if (flag) {
      int tag = st.MPI_TAG;
      switch (tag) {
      case TAG_NODE_STEAL_REQ:
        inter_node_work_steal_victim(availableWork, st, my_color, tok, comms);
        break;
      case TAG_TOKEN:
        receive_token(tok, st, have_token, comms);
        break;
      }
    }
    try_forward_token_if_idle(availableWork, have_token, my_color, tok,
                              next_leader, term_win, comms);
    // If leader node is idle, initiate a steal request
    if (length(availableWork) <= 0 && !(*global_done)) {
      inter_node_work_steal_initiate(availableWork, st, num_workers, my_color,
                                     tok, term_win, global_done, comms);
    }
  }
  // Poison the workers
  send_poison_pill(num_workers, comms);
  MPI_Win_unlock_all(term_win);
  MPI_Win_free(&term_win);
}

static void worker_hierarchical(int worker_local_rank, WorkChunk &myChunk,
                                double &localBestMaxF, int localComb[],
                                sets_t dataTable, SET *buffers,
                                double elapsed_times[], CommsStruct &comms) {

  while (true) {
    process_lambda_interval(myChunk.start, myChunk.end, localComb,
                            localBestMaxF, dataTable, buffers, elapsed_times,
                            comms);
    char dummy;
    MPI_Send(&dummy, 1, MPI_BYTE, 0, TAG_REQUEST_WORK, comms.local_comm);
    MPI_Status status;
    MPI_Recv(&myChunk, sizeof(WorkChunk), MPI_BYTE, 0, TAG_ASSIGN_WORK,
             comms.local_comm, &status);
    printf("[DEBUG Worker_hierarchical] Rank %d received chunk [%lld, %lld] "
           "from node leader\n",
           comms.global_rank, myChunk.start, myChunk.end);
    fflush(stdout);

    if (length(myChunk) < 0) {
      break;
    }
  }
}

static inline void
execute_hierarchical(int rank, int size_minus_one, LAMBDA_TYPE num_Comb,
                     double &localBestMaxF, int localComb[], sets_t dataTable,
                     SET *buffers, double elapsed_times[], CommsStruct &comms) {
  WorkChunk leaderRange = calculate_node_range(num_Comb, comms);
  const int num_workers = comms.local_size - 1;

  if (comms.is_leader) {
    leaderRange.start += (CHUNK_SIZE * num_workers);
    node_leader_hierarchical(leaderRange, num_workers, comms);
    printf("[DEBUG Execute hierarchical] Rank %d (Node %d, Local Rank %d): "
           "Assigned chunk [%lld, "
           "%lld]\n",
           comms.global_rank, comms.my_node_id, comms.local_rank,
           leaderRange.start, leaderRange.end);
    fflush(stdout);
  } else {
    const int worker_id = comms.local_rank - 1;
    WorkChunk myChunk =
        calculate_worker_range(leaderRange, worker_id, num_workers);
    printf("[DEBUG Execute hierarchical] Rank %d (Node %d, Local Rank %d): "
           "Assigned chunk [%lld, "
           "%lld]\n",
           comms.global_rank, comms.my_node_id, comms.local_rank, myChunk.start,
           myChunk.end);
    fflush(stdout);

    worker_hierarchical(comms.local_rank, myChunk, localBestMaxF, localComb,
                        dataTable, buffers, elapsed_times, comms);
  }
}

#else // Not using hierachical Allreduce

#define ALL_REDUCE_FUNC MPI_Allreduce
#define EXECUTE execute_role

#endif // ALL_REDUCE_HIERARCHICAL

//////////////////////////////  End Allreduce_hierarchical
/////////////////////////

static inline LAMBDA_TYPE calculate_initial_index(int num_workers) {
  return static_cast<LAMBDA_TYPE>(num_workers) * CHUNK_SIZE;
}

static inline void distribute_work(int num_workers, LAMBDA_TYPE num_Comb,
                                   LAMBDA_TYPE &next_idx) {
  while (next_idx < num_Comb) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      int workerRank = status.MPI_SOURCE;
      char message;
      MPI_Recv(&message, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);

      if (message == 'a') {
        MPI_Send(&next_idx, 1, MPI_LONG_LONG_INT, workerRank, 2,
                 MPI_COMM_WORLD);
        std::cout << "[At time epoch: " << MPI_Wtime() << "s] "
                  << "Sent combos [" << next_idx << "–"
                  << (std::min(next_idx + CHUNK_SIZE, num_Comb) - 1)
                  << "] to rank " << workerRank << ". "
                  << (num_Comb - std::min(next_idx + CHUNK_SIZE, num_Comb))
                  << " combos left." << std::endl;
        next_idx += CHUNK_SIZE;
      }
    }
  }
}

static inline void master_process(int num_workers, LAMBDA_TYPE num_Comb) {
  LAMBDA_TYPE next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);

  LAMBDA_TYPE termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

static inline LAMBDA_TYPE nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) = C(n, n-r)

  LAMBDA_TYPE result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

static void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

std::pair<LAMBDA_TYPE, LAMBDA_TYPE>
calculate_initial_chunk(int rank, LAMBDA_TYPE num_Comb,
                        LAMBDA_TYPE chunk_size) {
  LAMBDA_TYPE begin = (rank - 1) * chunk_size;
  LAMBDA_TYPE end = std::min(begin + chunk_size, num_Comb);
  return {begin, end};
}

static inline LambdaComputed compute_lambda_variables(LAMBDA_TYPE lambda,
                                                      int totalGenes) {
  LambdaComputed computed;
  computed.j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
  if (computed.j > totalGenes - (NUMHITS - 2)) {
    computed.j = -1;
    return computed;
  }
  computed.i = lambda - (computed.j * (computed.j - 1)) / 2;
  return computed;
}

static void write_output(int rank, std::ofstream &outfile,
                         const int globalBestComb[], double F_max) {
  outfile << "(";
  for (size_t idx = 0; idx < NUMHITS; ++idx) {
    outfile << globalBestComb[idx];
    if (idx != NUMHITS - 1) {
      outfile << ", ";
    }
  }
  outfile << ")  F-max = " << F_max << std::endl;
}

static void max_f_with_comb(void *in, void *inout, int *len,
                            MPI_Datatype *type) {
  const MPIResultWithComb *in_vals = static_cast<const MPIResultWithComb *>(in);
  MPIResultWithComb *inout_vals = static_cast<MPIResultWithComb *>(inout);

  for (int i = 0; i < *len; i++) {
    if (in_vals[i].f > inout_vals[i].f) {
      inout_vals[i].f = in_vals[i].f;
      for (int j = 0; j < NUMHITS; j++) {
        inout_vals[i].comb[j] = in_vals[i].comb[j];
      }
    }
  }
}

static MPI_Op create_max_f_with_comb_op(MPI_Datatype MPI_RESULT_WITH_COMB) {
  MPI_Op MPI_MAX_F_WITH_COMB;
  MPI_Op_create(&max_f_with_comb, 1, &MPI_MAX_F_WITH_COMB);
  return MPI_MAX_F_WITH_COMB;
}

static MPI_Datatype create_mpi_result_with_comb_type() {
  MPI_Datatype MPI_RESULT_WITH_COMB;

  const int nitems = 2;
  int blocklengths[] = {1, NUMHITS};
  MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT};

  MPI_Aint offsets[nitems];
  offsets[0] = offsetof(MPIResultWithComb, f);
  offsets[1] = offsetof(MPIResultWithComb, comb);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &MPI_RESULT_WITH_COMB);
  MPI_Type_commit(&MPI_RESULT_WITH_COMB);

  return MPI_RESULT_WITH_COMB;
}

static inline void process_lambda_interval(LAMBDA_TYPE startComb,
                                           LAMBDA_TYPE endComb,
                                           int bestCombination[], double &maxF,
                                           sets_t &dataTable, SET *buffers,
                                           double elapsed_times[],
                                           CommsStruct &comms) {

  const int totalGenes = dataTable.numRows;
  const double alpha = 0.1;
  int localComb[NUMHITS] = {0};

  for (LAMBDA_TYPE lambda = startComb; lambda <= endComb; ++lambda) {

    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
    if (lambda == startComb || lambda == startComb + 1) {
      printf("[DEBUG] Rank %d (node %d, local %d): lambda=%lld => i=%d, j=%d\n",
             comms.global_rank, comms.my_node_id, comms.local_rank, lambda,
             computed.i, computed.j);
      fflush(stdout);
    }
    if (computed.j < 0)
      continue;

    SET rowI =
        GET_ROW(dataTable.tumorData, computed.i, dataTable.tumorRowUnits);
    SET rowJ =
        GET_ROW(dataTable.tumorData, computed.j, dataTable.tumorRowUnits);
    SET_INTERSECT(buffers[0], rowI, rowJ, dataTable.tumorRowUnits);
    if (SET_IS_EMPTY(buffers[0], dataTable.tumorRowUnits))
      continue;

    localComb[0] = computed.i;
    localComb[1] = computed.j;

    int indices[NUMHITS];
    indices[0] = computed.i;
    indices[1] = computed.j;
    indices[2] = computed.j + 1;
    int level = 2;

    while (level >= 2) {
      int maxStart = totalGenes - (NUMHITS - (level + 1));
      if (indices[level] >= maxStart) {
        --level;
        if (level >= 2) {
          ++indices[level];
        }
        continue;
      }

      SET rowK =
          GET_ROW(dataTable.tumorData, indices[level], dataTable.tumorRowUnits);
      SET_INTERSECT(buffers[level - 1], buffers[level - 2], rowK,
                    dataTable.tumorRowUnits);
      if (SET_IS_EMPTY(buffers[level - 1], dataTable.tumorRowUnits)) {
        ++indices[level];
        continue;
      }

      localComb[level] = indices[level];

      if (level == NUMHITS - 1) {
        INCREMENT_COMBO_COUNT(elapsed_times);
        int TP = SET_COUNT(buffers[NUMHITS - 2], dataTable.tumorRowUnits);
        SET normalRows[NUMHITS];
        for (int idx = 0; idx < NUMHITS; ++idx) {
          normalRows[idx] = GET_ROW(dataTable.normalData, localComb[idx],
                                    dataTable.normalRowUnits);
        }
        SET_INTERSECT_N(buffers[NUMHITS - 2], normalRows, NUMHITS,
                        dataTable.normalRowUnits);
        int coveredNormal =
            SET_COUNT(buffers[NUMHITS - 2], dataTable.normalRowUnits);
        int TN = (int)dataTable.numNormal - coveredNormal;
        double F =
            (alpha * TP + TN) / (dataTable.numTumor + dataTable.numNormal);
        // printf("[rank %d] λ=%lld  TP=%d  TN=%d  F=%.5f\n", comms.local_rank,
        //        lambda, TP, TN, F);
        // fflush(stdout);
        if (F >= maxF) {
          maxF = F;
          for (int k = 0; k < NUMHITS; ++k)
            bestCombination[k] = localComb[k];
        }
        printf("[DEBUG] Rank %d local maxF=%.5f bestComb=[", comms.global_rank,
               maxF);
        fflush(stdout);

        for (int k = 0; k < NUMHITS; ++k)
          printf("%d ", bestCombination[k]);
        printf("]\n");
        fflush(stdout);

        ++indices[level];
      } else {
        ++level;
        indices[level] = indices[level - 1] + 1;
      }
    }
  }
}

static bool process_and_communicate(int rank, LAMBDA_TYPE num_Comb,
                                    double &localBestMaxF, int localComb[],
                                    LAMBDA_TYPE &begin, LAMBDA_TYPE &end,
                                    MPI_Status &status, sets_t dataTable,
                                    SET *buffers, double elapsed_times[],
                                    CommsStruct &comms) {
  START_TIMING(run_time);
  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          buffers, elapsed_times, comms);
  END_TIMING(run_time, elapsed_times[WORKER_RUNNING_TIME]);
  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
  LAMBDA_TYPE next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);

  if (next_idx == -1) {
    return false;
  }

  begin = next_idx;
  end = std::min(begin + CHUNK_SIZE, num_Comb);
  return true;
}

static void worker_process(int rank, LAMBDA_TYPE num_Comb,
                           double &localBestMaxF, int localComb[],
                           sets_t dataTable, SET *buffers,
                           double elapsed_times[], CommsStruct &comms) {
  std::pair<LAMBDA_TYPE, LAMBDA_TYPE> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);

  LAMBDA_TYPE begin = chunk_indices.first;
  LAMBDA_TYPE end = chunk_indices.second;

  MPI_Status status;

  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, localBestMaxF, localComb, begin, end, status, dataTable,
        buffers, elapsed_times, comms);
    if (!has_next) {
      break;
    }
  }
}

static void execute_role(int rank, int size_minus_one, LAMBDA_TYPE num_Comb,
                         double &localBestMaxF, int localComb[],
                         sets_t dataTable, SET *buffers, double elapsed_times[],
                         CommsStruct &comms) {
  if (rank == 0) {
    START_TIMING(master_proc);
    master_process(size_minus_one, num_Comb);
    END_TIMING(master_proc, elapsed_times[MASTER_TIME]);
  } else {
    START_TIMING(worker_proc);
    worker_process(rank, num_Comb, localBestMaxF, localComb, dataTable, buffers,
                   elapsed_times, comms);
    END_TIMING(worker_proc, elapsed_times[WORKER_TIME]);
  }
}

static inline void initialize_local_comb_and_f(double &f, int localComb[]) {
  f = 0;
  for (int i = 0; i < NUMHITS; ++i) {
    localComb[i] = -1;
  }
}

static MPIResultWithComb create_mpi_result(double f, const int comb[]) {
  MPIResultWithComb result;
  result.f = f;
  for (int i = 0; i < NUMHITS; ++i) {
    result.comb[i] = comb[i];
  }
  return result;
}

static void extract_global_comb(int globalBestComb[],
                                const MPIResultWithComb &globalResult) {
  for (int i = 0; i < NUMHITS; ++i) {
    globalBestComb[i] = globalResult.comb[i];
  }
}

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable,
                      CommsStruct &comms) {
  int Nt = dataTable.numTumor;
  int numGenes = dataTable.numRows;

  size_t tumorBits = dataTable.numTumor;
  size_t tumorUnits = dataTable.tumorRowUnits;
  size_t maxUnits = std::max(dataTable.tumorRowUnits, dataTable.normalRowUnits);
  SET buffers[NUMHITS - 1];

  for (int i = 0; i < NUMHITS - 1; i++) {
    SET_NEW(buffers[i], maxUnits);
  }

  MPI_Datatype MPI_RESULT_WITH_COMB = create_mpi_result_with_comb_type();
  MPI_Op MPI_MAX_F_WITH_COMB = create_max_f_with_comb_op(MPI_RESULT_WITH_COMB);

  LAMBDA_TYPE num_Comb = nCr(numGenes, 2);

  SET droppedSamples;
  SET_NEW(droppedSamples, tumorUnits);

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (
      !CHECK_ALL_BITS_SET(droppedSamples, tumorBits, dataTable.tumorRowUnits)) {
    double localBestMaxF;
    int localComb[NUMHITS];
    initialize_local_comb_and_f(localBestMaxF, localComb);

    EXECUTE(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable,
            buffers, elapsed_times, comms);

    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult = {};
    ALL_REDUCE_FUNC(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                    MPI_MAX_F_WITH_COMB, comms);
    int globalBestComb[NUMHITS];
    extract_global_comb(globalBestComb, globalResult);
    if (rank == 0) { // or all ranks, for cross-check
      printf("[DEBUG] After ALLREDUCE: globalBestComb=[");
      for (int k = 0; k < NUMHITS; ++k)
        printf("%d ", globalBestComb[k]);
      printf("] F=%.5f\n", globalResult.f);
      fflush(stdout);
    }
    SET intersectionSets[NUMHITS];
    for (int i = 0; i < NUMHITS; ++i) {
      intersectionSets[i] =
          GET_ROW(dataTable.tumorData, globalBestComb[i], tumorUnits);
    }

    SET_INTERSECT_N(buffers[NUMHITS - 2], intersectionSets, NUMHITS,
                    tumorUnits);

    SET_UNION(droppedSamples, droppedSamples, buffers[NUMHITS - 2],
              dataTable.tumorRowUnits);

    UPDATE_SET_COLLECTION(dataTable.tumorData, buffers[NUMHITS - 2],
                          dataTable.numRows, dataTable.tumorRowUnits);

    Nt -= SET_COUNT(buffers[NUMHITS - 2], dataTable.tumorRowUnits);

    if (rank == 0)
      write_output(rank, outfile, globalBestComb, globalResult.f);
  }

  if (rank == 0)
    outfile.close();

  for (int i = 0; i < NUMHITS - 1; i++) {
    SET_FREE(buffers[i]);
  }
  SET_FREE(droppedSamples);

  MPI_Op_free(&MPI_MAX_F_WITH_COMB);
  MPI_Type_free(&MPI_RESULT_WITH_COMB);
}
