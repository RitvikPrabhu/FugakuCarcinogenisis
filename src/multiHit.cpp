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
  const LAMBDA_TYPE nodeLen = leaderRange.end - leaderRange.start + 1;

  const LAMBDA_TYPE base = nodeLen / num_workers;
  const LAMBDA_TYPE extra = nodeLen % num_workers;

  const LAMBDA_TYPE start = leaderRange.start + worker_id * base +
                            std::min<LAMBDA_TYPE>(worker_id, extra);

  const LAMBDA_TYPE len = base + (worker_id < extra ? 1 : 0);

  return {start, start + len - 1};
}

inline LAMBDA_TYPE length(const WorkChunk &c) { return c.end - c.start + 1; }

inline static void handle_local_work_request(std::vector<WorkChunk> &table,
                                             MPI_Status st, int num_workers,
                                             int active_workers,
                                             const CommsStruct &comms) {

  printf("Starting local steal requests\n");
  fflush(stdout);
  int requester = st.MPI_SOURCE;
  char dummy;
  MPI_Recv(&dummy, 1, MPI_BYTE, requester, TAG_REQUEST_WORK, comms.local_comm,
           MPI_STATUS_IGNORE);

  int donor = -1;
  LAMBDA_TYPE bestLen = 0;
  for (int w = 1; w <= num_workers; ++w) {
    LAMBDA_TYPE len = length(table[w]);
    if (len > bestLen) {
      bestLen = len;
      donor = w;
    }
  }

  WorkChunk reply{0, -1};
  if (donor != -1 && bestLen > 1) {
    WorkChunk &d = table[donor];
    LAMBDA_TYPE mid = d.start + (bestLen / 2) - 1;
    reply.start = mid + 1;
    reply.end = d.end;
    d.end = mid;
    LAMBDA_TYPE tmpEnd = d.end;
    MPI_Request rq;
    MPI_Isend(&tmpEnd, 1, MPI_LONG_LONG_INT, donor, TAG_UPDATE_END,
              comms.local_comm, &rq);
    ++active_workers;
  } else {
    --active_workers;
  }
  MPI_Request rq;
  MPI_Isend(&reply, sizeof(WorkChunk), MPI_BYTE, requester, TAG_ASSIGN_WORK,
            comms.local_comm, &rq);

  if (length(reply) > 0)
    table[requester] = reply;
}

inline static void worker_progress_update(std::vector<WorkChunk> &table,
                                          MPI_Status st,
                                          const CommsStruct &comms) {
  printf("Starting per-worker progress harvest\n");
  fflush(stdout);
  LAMBDA_TYPE newStart;
  MPI_Request rq;
  MPI_Irecv(&newStart, 1, MPI_LONG_LONG_INT, st.MPI_SOURCE, TAG_UPDATE_START,
            comms.local_comm, &rq);
  table[st.MPI_SOURCE].start = newStart;
}
static void node_leader_hierarchical(const WorkChunk &leaderRange,
                                     int num_workers,
                                     const CommsStruct &comms) {
  printf("In node leader\n");
  fflush(stdout);
  std::vector<WorkChunk> table(num_workers + 1);
  for (int w = 1; w <= num_workers; ++w)
    table[w] = calculate_worker_range(leaderRange, w - 1, num_workers);

  int active_workers = num_workers;
  bool global_done = false;
  while (!global_done) {
    MPI_Status st;
    int flag;
    // local probe
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms.local_comm, &flag, &st);
    if (flag) {
      int source = st.MPI_SOURCE;
      int tag = st.MPI_TAG;
      switch (tag) {
      case TAG_REQUEST_WORK:
        handle_local_work_request(table, st, num_workers, active_workers,
                                  comms);
      case TAG_UPDATE_START:
        worker_progress_update(table, st, comms);
      }
    }

    // global probe
    flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms.global_comm, &flag, &st);
    if (flag) {
      int tag = st.MPI_TAG;
      switch (tag) {
      case TAG_NODE_STEAL_REQ:
        break;
      case TAG_NODE_STEAL_REPLY:
        break;
      }
    }
  }
}
/**
static void node_leader_hierarchical(const WorkChunk &leaderRange,
                                     int num_workers,
                                     const CommsStruct &comms) {
  printf("In node leader\n");
  fflush(stdout);
  std::vector<WorkChunk> table(num_workers + 1);
  for (int w = 1; w <= num_workers; ++w)
    table[w] = calculate_worker_range(leaderRange, w - 1, num_workers);

  int active_workers = num_workers;
  bool global_done = false;
  while (!global_done) {

    // harvest per-worker progress hints
    MPI_Status stStart;
    int flagStart = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_UPDATE_START, comms.local_comm, &flagStart,
               &stStart);
    if (flagStart) {
      printf("Starting per-worker progress harvest\n");
      fflush(stdout);
      LAMBDA_TYPE newStart;
      MPI_Recv(&newStart, 1, MPI_LONG_LONG_INT, stStart.MPI_SOURCE,
               TAG_UPDATE_START, comms.local_comm, MPI_STATUS_IGNORE);
      table[stStart.MPI_SOURCE].start = newStart;
    }

    // answer inter-node steal requests
    MPI_Status stNode;
    int flagNode = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_NODE_STEAL_REQ, comms.global_comm, &flagNode,
               &stNode);
    if (flagNode) {
      printf("Starting inter-node steal requests\n");
      fflush(stdout);
      char dummy;
      MPI_Recv(&dummy, 1, MPI_BYTE, stNode.MPI_SOURCE, TAG_NODE_STEAL_REQ,
               comms.global_comm, MPI_STATUS_IGNORE);

      int donor = -1;
      LAMBDA_TYPE bestLen = 0;
      for (int w = 1; w <= num_workers; ++w) {
        LAMBDA_TYPE len = length(table[w]);
        if (len > bestLen) {
          bestLen = len;
          donor = w;
        }
      }

      WorkChunk reply{0, -1};
      if (donor != -1 && bestLen > 0) {
        reply = table[donor];
        table[donor] = {0, -1};
        LAMBDA_TYPE newEnd = table[donor].start - 1;
        table[donor].end = newEnd;
        LAMBDA_TYPE tmpEnd = newEnd;
        fprintf(stderr, "[%d] TAG_UPDATE_END buffer %p\n", comms.global_rank,
                (void *)&tmpEnd);
        fflush(stderr);
        MPI_Send(&tmpEnd, 1, MPI_LONG_LONG_INT, donor, TAG_UPDATE_END,
                 comms.local_comm);
      }
      MPI_Send(&reply, sizeof(WorkChunk), MPI_BYTE, stNode.MPI_SOURCE,
               TAG_NODE_STEAL_REPLY, comms.global_comm);
    }

   // local work-steal requests
    MPI_Status stReq;
    int flagReq = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST_WORK, comms.local_comm, &flagReq,
               &stReq);
    if (flagReq) {
      printf("Starting local steal requests\n");
      fflush(stdout);
      int requester = stReq.MPI_SOURCE;
      char dummy;
      MPI_Recv(&dummy, 1, MPI_BYTE, requester, TAG_REQUEST_WORK,
               comms.local_comm, MPI_STATUS_IGNORE);

      int donor = -1;
      LAMBDA_TYPE bestLen = 0;
      for (int w = 1; w <= num_workers; ++w) {
        LAMBDA_TYPE len = length(table[w]);
        if (len > bestLen) {
          bestLen = len;
          donor = w;
        }
      }

      WorkChunk reply{0, -1};
      if (donor != -1 && bestLen > 1) {
        WorkChunk &d = table[donor];
        LAMBDA_TYPE mid = d.start + (bestLen / 2) - 1;
        reply.start = mid + 1;
        reply.end = d.end;
        d.end = mid;
        LAMBDA_TYPE tmpEnd = d.end;
        MPI_Send(&tmpEnd, 1, MPI_LONG_LONG_INT, donor, TAG_UPDATE_END,
                 comms.local_comm);
        ++active_workers;
      } else {
        --active_workers;
      }

      MPI_Send(&reply, sizeof(WorkChunk), MPI_BYTE, requester, TAG_ASSIGN_WORK,
               comms.local_comm);

      if (length(reply) > 0)
        table[requester] = reply;
    }

    // if entire node idle, become thief
    if (active_workers == 0) {
      printf("Becoming theif\n");
      fflush(stdout);
      int myRank, nLeaders;
      MPI_Comm_rank(comms.global_comm, &myRank);
      MPI_Comm_size(comms.global_comm, &nLeaders);

      bool lootReceived = false;
      char dummy;

      for (int attempt = 0; attempt < 3 && !lootReceived; ++attempt) {
        int victim;
        do {
          victim = rand() % nLeaders;
        } while (victim == myRank);
        MPI_Send(&dummy, 1, MPI_BYTE, victim, TAG_NODE_STEAL_REQ,
                 comms.global_comm);

        WorkChunk loot;
        MPI_Recv(&loot, sizeof(WorkChunk), MPI_BYTE, victim,
                 TAG_NODE_STEAL_REPLY, comms.global_comm, MPI_STATUS_IGNORE);

        if (length(loot) > 0) {
          printf("Rank %d received work from %d\n", comms.global_rank, victim);
          fflush(stdout);

          for (int w = 1; w <= num_workers; ++w)
            table[w] = calculate_worker_range(loot, w - 1, num_workers);

          for (int w = 1; w <= num_workers; ++w) {
            MPI_Send(&table[w], sizeof(WorkChunk), MPI_BYTE, w, TAG_ASSIGN_WORK,
                     comms.local_comm);
          }
          active_workers = num_workers;
          lootReceived = true;
        }
      }
      if (!lootReceived) {
        printf("Should we kill the job?\n");
        fflush(stdout);

        int localDone = 1;
        int globalDone = 0;
        printf("Before all reduce\n");
        fflush(stdout);
        MPI_Allreduce(&localDone, &globalDone, 1, MPI_INT, MPI_LAND,
                      comms.global_comm);
        printf("After all reduce\n");
        fflush(stdout);
        if (globalDone) {
          WorkChunk empty{0, -1};
          for (int w = 1; w <= num_workers; ++w) {
            MPI_Send(&empty, sizeof(WorkChunk), MPI_BYTE, w, TAG_ASSIGN_WORK,
                     comms.local_comm);
          }
          global_done = true;
        }
      } else {
        active_workers = num_workers;
      }
    }
  }
}**/

static void worker_hierarchical(int worker_local_rank, WorkChunk &myChunk,
                                double &localBestMaxF, int localComb[],
                                sets_t dataTable, SET *buffers,
                                double elapsed_times[], CommsStruct &comms) {

  printf("Rank %d inside worker hierarchical\n", comms.local_rank);
  fflush(stdout);
  while (true) {
    process_lambda_interval(myChunk.start, myChunk.end, localComb,
                            localBestMaxF, dataTable, buffers, elapsed_times,
                            comms);
    printf("Rank %d has finished a round of process_lambda_interval\n",
           comms.local_rank);
    fflush(stdout);
    char dummy;
    MPI_Send(&dummy, 1, MPI_BYTE, 0, TAG_REQUEST_WORK, comms.local_comm);
    MPI_Request rq;
    MPI_Irecv(&myChunk, sizeof(WorkChunk), MPI_BYTE, 0, TAG_ASSIGN_WORK,
              comms.local_comm, &rq);

    if (length(myChunk) == 0)
      break;
  }
}

static inline void
execute_hierarchical(int rank, int size_minus_one, LAMBDA_TYPE num_Comb,
                     double &localBestMaxF, int localComb[], sets_t dataTable,
                     SET *buffers, double elapsed_times[], CommsStruct &comms) {
  WorkChunk leaderRange = calculate_node_range(num_Comb, comms);
  const int num_workers = comms.local_size - 1;

  WorkChunk myChunk;
  if (comms.local_rank == 0) {
    myChunk = leaderRange;
  } else {
    const int worker_id = comms.local_rank - 1;
    myChunk = calculate_worker_range(leaderRange, worker_id, num_workers);
  }

  /* Leader stores everybody’s initial allocation for bookkeeping ---- */
  static std::vector<WorkChunk> initialMap;
  if (comms.is_leader) {
    initialMap.resize(num_workers);
    for (int w = 0; w < num_workers; ++w)
      initialMap[w] = calculate_worker_range(leaderRange, w, num_workers);
  }
  if (comms.local_rank == 0) {
    node_leader_hierarchical(leaderRange, num_workers, comms);
  } else {
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

static inline bool check_for_assignment(LAMBDA_TYPE &endComb,
                                        MPI_Comm local_comm) {
  MPI_Status st;
  int flag = 0;
  MPI_Iprobe(0, TAG_UPDATE_END, local_comm, &flag, &st);
  if (!flag)
    return false;

  LAMBDA_TYPE newEnd;
  MPI_Request rq;
  MPI_Irecv(&newEnd, 1, MPI_LONG_LONG_INT, 0, TAG_UPDATE_END, local_comm, &rq);

  endComb = newEnd;
  return true;
}

static inline void process_lambda_interval(LAMBDA_TYPE startComb,
                                           LAMBDA_TYPE endComb,
                                           int bestCombination[], double &maxF,
                                           sets_t &dataTable, SET *buffers,
                                           double elapsed_times[],
                                           CommsStruct &comms) {

  printf("Starting process_lambda_interval\n");
  fflush(stdout);
  const int totalGenes = dataTable.numRows;
  const double alpha = 0.1;
  int localComb[NUMHITS] = {0};

  for (LAMBDA_TYPE lambda = startComb; lambda <= endComb; ++lambda) {
    printf("Lambda loop, at lambda = %lld out of %lld\n", lambda, endComb);
    fflush(stdout);
#ifdef HIERARCHICAL_COMMS
    check_for_assignment(endComb, comms.local_comm);
    if (lambda > endComb)
      break;
    LAMBDA_TYPE start = lambda;
    MPI_Request rq;
    MPI_Isend(&start, 1, MPI_LONG_LONG_INT, 0, TAG_UPDATE_START,
              comms.local_comm, &rq);
#endif

    printf("Finished hierarchical comms\n");
    fflush(stdout);
    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
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
        if (F >= maxF) {
          maxF = F;
          for (int k = 0; k < NUMHITS; ++k)
            bestCombination[k] = localComb[k];
        }
        ++indices[level];
      } else {
        ++level;
        indices[level] = indices[level - 1] + 1;
      }
    }
    /**
    printf("Starting check_for_assignment\n");
    fflush(stdout);
    #ifdef HIERARCHICAL_COMMS
        check_for_assignment(endComb, comms.local_comm);
    #endif
    printf("Finished check for assignment\n");
    fflush(stdout);**/
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
  printf("Thisis the latest version 2\n");
  fflush(stdout);
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
