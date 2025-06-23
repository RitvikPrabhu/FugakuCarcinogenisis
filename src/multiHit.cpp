#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstring>
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

static inline void execute_hierarchical(int rank, int size_minus_one,
                                        LAMBDA_TYPE num_Comb,
                                        double &localBestMaxF, int localComb[],
                                        sets_t dataTable, SET *buffers,
                                        double elapsed_times[],
                                        const CommsStruct &comms) {
  WorkChunk leaderRange = calculate_node_range(num_Comb, comms);
  const int num_workers = comms.local_size - 1;

  WorkChunk myChunk;
  if (comms.local_rank == 0) { // I'm the node-leader
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
                                           double elapsed_times[]) {

  const int totalGenes = dataTable.numRows;
  const double alpha = 0.1;
  int localComb[NUMHITS] = {0};

  for (LAMBDA_TYPE lambda = startComb; lambda <= endComb; ++lambda) {
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
  }
}

static bool process_and_communicate(int rank, LAMBDA_TYPE num_Comb,
                                    double &localBestMaxF, int localComb[],
                                    LAMBDA_TYPE &begin, LAMBDA_TYPE &end,
                                    MPI_Status &status, sets_t dataTable,
                                    SET *buffers, double elapsed_times[]) {
  START_TIMING(run_time);
  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          buffers, elapsed_times);
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
                           double elapsed_times[]) {
  std::pair<LAMBDA_TYPE, LAMBDA_TYPE> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);

  LAMBDA_TYPE begin = chunk_indices.first;
  LAMBDA_TYPE end = chunk_indices.second;

  MPI_Status status;

  while (end <= num_Comb) {
    bool has_next =
        process_and_communicate(rank, num_Comb, localBestMaxF, localComb, begin,
                                end, status, dataTable, buffers, elapsed_times);
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
                   elapsed_times);
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
    /**
        MPIResultWithComb localResult = create_mpi_result(localBestMaxF,
       localComb); MPIResultWithComb globalResult = {};
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
          write_output(rank, outfile, globalBestComb, globalResult.f);**/
    break;
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
