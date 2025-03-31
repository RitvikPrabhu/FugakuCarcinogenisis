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

#include "fourHit.h"
#include "utils.h"
/// Here chose to uncomment one of these lines to
// switch between hierarchical or vanilla MPI_Allreduce function
#define ALL_REDUCE_HIERARCHICAL 1
//#undef ALL_REDUCE_HIERARCHICAL

//////////////////////////////  Start Allreduce_hierarchical
/////////////////////////

#ifdef ALL_REDUCE_HIERARCHICAL
#include <unistd.h>

#define ALL_REDUCE_FUNC Allreduce_hierarchical
#define MAX_NAME_LEN 256

int hash_hostname(const char *hostname) {
  int hash = 0;
  while (*hostname) {
    hash = (hash * 31) ^ (*hostname); // Prime-based hashing with XOR
    hostname++;
  }
  return hash & 0x7FFFFFFF; // Ensure positive value
}

void Allreduce_hierarchical(void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

  static MPI_Comm local_comm = MPI_COMM_NULL;
  static MPI_Comm global_comm = MPI_COMM_NULL;
  static int local_rank, local_size;
  static int world_rank, world_size;
  static int global_rank = -1; // Only valid in global_comm

  if (local_comm == MPI_COMM_NULL) { // not Already initialized

    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // Get node name
    char node_name[MAX_NAME_LEN];
    gethostname(node_name, MAX_NAME_LEN);

    // Generate a unique color per node (hashing the hostname)
    int node_color = hash_hostname(node_name);
    /* int node_color = extract_number_from_hostname(node_name); */
    //printf("[%d] name: %s. Color: %d\n", world_rank, node_name, node_color);

    // Create local communicator
    MPI_Comm_split(comm, node_color, world_rank, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    // Rank 0 of each local communicator joins the global communicator
    int global_color = (local_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(comm, global_color, world_rank, &global_comm);

    // Get global rank (if in global_comm)
    if (local_rank == 0) {
      MPI_Comm_rank(global_comm, &global_rank);
    }
  }

  // Datatype size
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);

  // Allocate buffer for local reduction
  void *local_result = malloc(count * datatype_size);

  // Phase 1: Local Reduction
  MPI_Reduce(sendbuf, local_result, count, datatype, op, 0, local_comm);

  // Allocate buffer for global reduction (only needed for rank 0 in local_comm)
  void *global_result = NULL;
  if (local_rank == 0) {
    global_result = malloc(count * datatype_size);
  }

  // Phase 2: Global Reduction (only rank 0 in each node participates)
  if (local_rank == 0) {
    MPI_Allreduce(local_result, global_result, count, datatype, op,
                  global_comm);
    // Copy the final result to recvbuf
    memcpy(recvbuf, global_result, count * datatype_size);
  }

  // Phase 3: Broadcast result to all processes in the local communicator
  MPI_Bcast(recvbuf, count, datatype, 0, local_comm);

  // Cleanup
  free(local_result);
  if (local_rank == 0)
    free(global_result);
}

#else // Not using hierachical Allreduce

#define ALL_REDUCE_FUNC MPI_Allreduce

#endif // ALL_REDUCE_HIERARCHICAL

//////////////////////////////  End Allreduce_hierarchical
/////////////////////////

inline LAMBDA_TYPE calculate_initial_index(int num_workers) {
  return static_cast<LAMBDA_TYPE>(num_workers) * CHUNK_SIZE;
}

inline void distribute_work(int num_workers, LAMBDA_TYPE num_Comb,
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
        next_idx += CHUNK_SIZE;
      }
    }
  }
}

inline void master_process(int num_workers, LAMBDA_TYPE num_Comb) {
  LAMBDA_TYPE next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);

  LAMBDA_TYPE termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

inline LAMBDA_TYPE nCr(int n, int r) {
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

void outputFileWriteError(std::ofstream &outfile) {

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

inline LambdaComputed compute_lambda_variables(LAMBDA_TYPE lambda,
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

void write_output(int rank, std::ofstream &outfile,
                  const std::array<int, NUMHITS> &globalBestComb,
                  double F_max) {
  outfile << "(";
  for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
    outfile << globalBestComb[idx];
    if (idx != globalBestComb.size() - 1) {
      outfile << ", ";
    }
  }
  outfile << ")  F-max = " << F_max << std::endl;
}

void max_f_with_comb(void *in, void *inout, int *len, MPI_Datatype *type) {
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

MPI_Op create_max_f_with_comb_op(MPI_Datatype MPI_RESULT_WITH_COMB) {
  MPI_Op MPI_MAX_F_WITH_COMB;
  MPI_Op_create(&max_f_with_comb, 1, &MPI_MAX_F_WITH_COMB);
  return MPI_MAX_F_WITH_COMB;
}

MPI_Datatype create_mpi_result_with_comb_type() {
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

void process_lambda_interval(LAMBDA_TYPE startComb, LAMBDA_TYPE endComb,
                             std::array<int, NUMHITS> &bestCombination,
                             double &maxF, sets_t &dataTable,
                             SET &intersectionBuffer, SET &scratchBufferij,
                             SET &scratchBufferijk, double elapsed_times[]) {
  double alpha = 0.1;
  int totalGenes = dataTable.numRows;

  for (LAMBDA_TYPE lambda = startComb; lambda <= endComb; lambda++) {
    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
    if (computed.j < 0) {
      continue;
    }
    START_TIMING(proc_row_ij);
    SET rowI =
        GET_ROW(dataTable.tumorData, computed.i, dataTable.tumorRowUnits);
    SET rowJ =
        GET_ROW(dataTable.tumorData, computed.j, dataTable.tumorRowUnits);
    END_TIMING(proc_row_ij, elapsed_times[PROCESS_LAMBDA_SET_COUNT]);

    START_TIMING(proc_intersect_ij);
    SET_INTERSECT(scratchBufferij, rowI, rowJ, dataTable.tumorRowUnits);
    END_TIMING(proc_row_ij, elapsed_times[PROCESS_LAMBDA_INTERSECT]);

    if (SET_IS_EMPTY(scratchBufferij, dataTable.tumorRowUnits)) {
      continue;
    }

    for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {
      START_TIMING(proc_row_k);
      SET rowK = GET_ROW(dataTable.tumorData, k, dataTable.tumorRowUnits);
      END_TIMING(proc_row_k, elapsed_times[PROCESS_LAMBDA_SET_COUNT]);

      START_TIMING(proc_intersect_ijk);
      SET_INTERSECT(scratchBufferijk, scratchBufferij, rowK,
                    dataTable.tumorRowUnits);
      END_TIMING(proc_intersect_ijk, elapsed_times[PROCESS_LAMBDA_INTERSECT]);

      if (SET_IS_EMPTY(scratchBufferijk, dataTable.tumorRowUnits)) {
        continue;
      }

      for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
        START_TIMING(proc_row_l);
        SET rowL = GET_ROW(dataTable.tumorData, l, dataTable.tumorRowUnits);
        END_TIMING(proc_row_l, elapsed_times[PROCESS_LAMBDA_SET_COUNT]);

        START_TIMING(proc_intersect_ijkl);
        SET_INTERSECT(intersectionBuffer, scratchBufferijk, rowL,
                      dataTable.tumorRowUnits);
        END_TIMING(proc_intersect_ijkl,
                   elapsed_times[PROCESS_LAMBDA_INTERSECT]);
        if (SET_IS_EMPTY(intersectionBuffer, dataTable.tumorRowUnits)) {
          continue;
        }
        INCREMENT_COMBO_COUNT(elapsed_times);

        START_TIMING(proc_count_TP);
        int TP = SET_COUNT(intersectionBuffer, dataTable.tumorRowUnits);
        END_TIMING(proc_count_TP, elapsed_times[PROCESS_LAMBDA_SET_COUNT]);

        START_TIMING(proc_row_normal);
        SET rowIN =
            GET_ROW(dataTable.normalData, computed.i, dataTable.normalRowUnits);
        SET rowJN =
            GET_ROW(dataTable.normalData, computed.j, dataTable.normalRowUnits);
        SET rowKN = GET_ROW(dataTable.normalData, k, dataTable.normalRowUnits);
        SET rowLN = GET_ROW(dataTable.normalData, l, dataTable.normalRowUnits);
        END_TIMING(proc_row_normal, elapsed_times[PROCESS_LAMBDA_SET_COUNT]);

        START_TIMING(proc_intersect_normal);
        SET_INTERSECT4(intersectionBuffer, rowIN, rowJN, rowKN, rowLN,
                       dataTable.normalRowUnits);
        END_TIMING(proc_intersect_normal,
                   elapsed_times[PROCESS_LAMBDA_INTERSECT]);

        START_TIMING(proc_count_coveredNormal);
        int coveredNormal =
            SET_COUNT(intersectionBuffer, dataTable.normalRowUnits);
        END_TIMING(proc_count_coveredNormal,
                   elapsed_times[PROCESS_LAMBDA_SET_COUNT]);
        int TN = (int)dataTable.numNormal - coveredNormal;
        double F =
            (alpha * TP + TN) / (dataTable.numTumor + dataTable.numNormal);
        if (F >= maxF) {
          maxF = F;
          bestCombination = {computed.i, computed.j, k, l};
        }
      }
    }
  }
}

bool process_and_communicate(int rank, LAMBDA_TYPE num_Comb,
                             double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb,
                             LAMBDA_TYPE &begin, LAMBDA_TYPE &end,
                             MPI_Status &status, sets_t dataTable,
                             SET &intersectionBuffer, SET &scratchBufferij,
                             SET &scratchBufferijk, double elapsed_times[]) {
  START_TIMING(run_time);
  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          intersectionBuffer, scratchBufferij, scratchBufferijk,
                          elapsed_times);
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

void worker_process(int rank, LAMBDA_TYPE num_Comb, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb, sets_t dataTable,
                    SET &intersectionBuffer, SET &scratchBufferij,
                    SET &scratchBufferijk, double elapsed_times[]) {
  std::pair<LAMBDA_TYPE, LAMBDA_TYPE> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);

  LAMBDA_TYPE begin = chunk_indices.first;
  LAMBDA_TYPE end = chunk_indices.second;

  MPI_Status status;

  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, localBestMaxF, localComb, begin, end, status, dataTable,
        intersectionBuffer, scratchBufferij, scratchBufferijk, elapsed_times);
    if (!has_next) {
      break;
    }
  }
}

void execute_role(int rank, int size_minus_one, LAMBDA_TYPE num_Comb,
                  double &localBestMaxF, std::array<int, NUMHITS> &localComb,
                  sets_t dataTable, SET &intersectionBuffer,
                  SET &scratchBufferij, SET &scratchBufferijk,
                  double elapsed_times[]) {
  if (rank == 0) {
    START_TIMING(master_proc);
    master_process(size_minus_one, num_Comb);
    END_TIMING(master_proc, elapsed_times[MASTER_TIME]);
  } else {
    START_TIMING(worker_proc);
    worker_process(rank, num_Comb, localBestMaxF, localComb, dataTable,
                   intersectionBuffer, scratchBufferij, scratchBufferijk,
                   elapsed_times);
    END_TIMING(worker_proc, elapsed_times[WORKER_TIME]);
  }
}

std::array<int, NUMHITS> initialize_local_comb_and_f(double &f) {
  f = 0;
  std::array<int, NUMHITS> comb;
  comb.fill(-1);
  return comb;
}

MPIResultWithComb create_mpi_result(double f,
                                    const std::array<int, NUMHITS> &comb) {
  MPIResultWithComb result;
  result.f = f;
  for (int i = 0; i < NUMHITS; ++i) {
    result.comb[i] = comb[i];
  }
  return result;
}

std::array<int, NUMHITS>
extract_global_comb(const MPIResultWithComb &globalResult) {
  std::array<int, NUMHITS> globalBestComb;
  for (int i = 0; i < NUMHITS; ++i) {
    globalBestComb[i] = globalResult.comb[i];
  }
  return globalBestComb;
}

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable) {

  int Nt = dataTable.numTumor;
  int numGenes = dataTable.numRows;

  size_t tumorBits = dataTable.numTumor;
  size_t tumorUnits = dataTable.tumorRowUnits;
  size_t maxUnits = std::max(dataTable.tumorRowUnits, dataTable.normalRowUnits);

  SET intersectionBuffer, scratchBufferij, scratchBufferijk;
  SET_NEW(intersectionBuffer, maxUnits);
  SET_NEW(scratchBufferij, maxUnits);
  SET_NEW(scratchBufferijk, maxUnits);

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
    std::array<int, NUMHITS> localComb =
        initialize_local_comb_and_f(localBestMaxF);

    START_TIMING(er_allreduce);
    START_TIMING(dist_er);
    execute_role(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable,
                 intersectionBuffer, scratchBufferij, scratchBufferijk,
                 elapsed_times);
    END_TIMING(dist_er, elapsed_times[DIST_ER]);

    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult = {};
    START_TIMING(dist_allreduce);
    ALL_REDUCE_FUNC(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                    MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);
    END_TIMING(er_allreduce, elapsed_times[ER_ALLREDUCE]);
	//idle = ER_ALLREDUCE - DIST_ER
    END_TIMING(dist_allreduce, elapsed_times[DIST_ALLREDUCE_TIME]);
    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);

    START_TIMING(dist_set_intersect);
    SET_INTERSECT4(intersectionBuffer,
                   GET_ROW(dataTable.tumorData, globalBestComb[0], tumorUnits),
                   GET_ROW(dataTable.tumorData, globalBestComb[1], tumorUnits),
                   GET_ROW(dataTable.tumorData, globalBestComb[2], tumorUnits),
                   GET_ROW(dataTable.tumorData, globalBestComb[3], tumorUnits),
                   dataTable.tumorRowUnits);
    END_TIMING(dist_set_intersect, elapsed_times[DIST_SET_INTERSECT_TIME]);

    START_TIMING(dist_set_union);
    SET_UNION(droppedSamples, droppedSamples, intersectionBuffer,
              dataTable.tumorRowUnits);
    END_TIMING(dist_set_union, elapsed_times[DIST_SET_UNION_TIME]);

    START_TIMING(dist_update_coll);
    UPDATE_SET_COLLECTION(dataTable.tumorData, intersectionBuffer,
                          dataTable.numRows, dataTable.tumorRowUnits);
    END_TIMING(dist_update_coll, elapsed_times[DIST_UPDATE_COLLECTION_TIME]);

    START_TIMING(dist_set_count);
    Nt -= SET_COUNT(intersectionBuffer, dataTable.tumorRowUnits);
    END_TIMING(dist_set_count, elapsed_times[DIST_SET_COUNT_TIME]);

    if (rank == 0)
      write_output(rank, outfile, globalBestComb, globalResult.f);
  }

  if (rank == 0)
    outfile.close();

  SET_FREE(intersectionBuffer);
  SET_FREE(scratchBufferij);
  SET_FREE(scratchBufferijk);
  SET_FREE(droppedSamples);

  MPI_Op_free(&MPI_MAX_F_WITH_COMB);
  MPI_Type_free(&MPI_RESULT_WITH_COMB);
}
