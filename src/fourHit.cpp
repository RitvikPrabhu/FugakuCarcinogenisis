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

/// Here chose to uncomment one of these lines to
// switch between hierarchical or vanilla MPI_Allreduce function 
#define ALL_REDUCE_HIERARCHICAL 1
// #undef  ALL_REDUCE_HIERARCHICAL


//////////////////////////////  Start Allreduce_hierarchical  //////////////////////
#ifdef ALL_REDUCE_HIERARCHICAL

#include <unistd.h>
#define MAX_NAME_LEN 256

int hash_hostname(const char *hostname) {
    int hash = 0;
    while (*hostname) {
        hash = (hash * 31) ^ (*hostname); // Prime-based hashing with XOR
        hostname++;
    }
    return hash & 0x7FFFFFFF;  // Ensure positive value
}

void Allreduce_hierarchical(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
				MPI_Op op, MPI_Comm comm) {

  static MPI_Comm local_comm = MPI_COMM_NULL;
  static MPI_Comm global_comm = MPI_COMM_NULL;
  static int local_rank, local_size;
  static int world_rank, world_size;
  static int global_rank = -1; // Only valid in global_comm

  
  if (local_comm == MPI_COMM_NULL){ // not Already initialized

    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // Get node name
    char node_name[MAX_NAME_LEN];
    gethostname(node_name, MAX_NAME_LEN);

    // Generate a unique color per node (hashing the hostname)
    int node_color = hash_hostname(node_name);
    /* int node_color = extract_number_from_hostname(node_name); */
    printf("[%d] name: %s. Color: %d\n",world_rank, node_name, node_color);
  
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
    MPI_Allreduce(local_result, global_result, count, datatype, op, global_comm);
    // Copy the final result to recvbuf
    memcpy(recvbuf, global_result, count * datatype_size);
  }

  // Phase 3: Broadcast result to all processes in the local communicator
  MPI_Bcast(recvbuf, count, datatype, 0, local_comm);

  // Cleanup
  free(local_result);
  if (local_rank == 0) free(global_result);
}
#endif //ALL_REDUCE_HIERARCHICAL


//////////////////////////////  End Allreduce_hierarchical //////////////////////







unit_t calculate_initial_index(int num_workers) {
  return num_workers * CHUNK_SIZE;
}

void distribute_work(int num_workers, unit_t num_Comb, unit_t &next_idx) {

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

void master_process(int num_workers, unit_t num_Comb) {
  unit_t next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);
  unit_t termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

inline unit_t nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) == C(n, n-r)

  unit_t result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

bool arrays_equal(const unit_t *a, const unit_t *b, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

double compute_F(int TP, int TN, double alpha, int Nt, int Nn) {
  return (alpha * TP + TN) / (Nt + Nn);
}

void update_tumor_data(unit_t *&tumorData, unit_t *sampleToCover, size_t units,
                       int numGenes) {
  for (int gene = 0; gene < numGenes; ++gene) {
    unit_t *geneRow = tumorData + gene * units;
    for (size_t i = 0; i < units; ++i) {
      geneRow[i] &= ~sampleToCover[i];
    }
  }
}

void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

std::pair<unit_t, unit_t> calculate_initial_chunk(int rank, unit_t num_Comb,
                                                  unit_t chunk_size) {
  unit_t begin = (rank - 1) * chunk_size;
  unit_t end = std::min(begin + chunk_size, num_Comb);
  return {begin, end};
}

void update_dropped_samples(unit_t *&droppedSamples, unit_t *sampleToCover,
                            size_t units) {
  for (size_t i = 0; i < units; ++i) {
    droppedSamples[i] |= sampleToCover[i];
  }
}

unit_t *initialize_dropped_samples(size_t units) {
  unit_t *droppedSamples = new unit_t[units];
  memset(droppedSamples, 0, units * sizeof(unit_t));
  return droppedSamples;
}

LambdaComputed compute_lambda_variables(unit_t lambda, int totalGenes) {
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

void process_lambda_interval(unit_t startComb, unit_t endComb,
                             std::array<int, NUMHITS> &bestCombination,
                             double &maxF, sets_t dataTable,
                             unit_t *&intersectionBuffer,
                             unit_t *&scratchBufferij,
                             unit_t *&scratchBufferijk) {
  double alpha = 0.1;
  size_t tumorUnits = UNITS_FOR_BITS(dataTable.numTumor);
  size_t normalUnits = UNITS_FOR_BITS(dataTable.numNormal);
  int totalGenes = dataTable.numRows;

  for (unit_t lambda = startComb; lambda <= endComb; lambda++) {
    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);

    unit_t *rowI = dataTable.tumorData + computed.i * tumorUnits;
    unit_t *rowJ = dataTable.tumorData + computed.j * tumorUnits;

    intersect_two_rows(scratchBufferij, rowI, rowJ, tumorUnits);

    if (is_empty(scratchBufferij, dataTable.numTumor))
      continue;

    for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {

      unit_t *rowK = dataTable.tumorData + k * tumorUnits;
      intersect_two_rows(scratchBufferijk, scratchBufferij, rowK, tumorUnits);

      if (is_empty(scratchBufferijk, dataTable.numTumor))
        continue;

      for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
        unit_t *rowL = dataTable.tumorData + l * tumorUnits;
        intersect_two_rows(intersectionBuffer, scratchBufferijk, rowL,
                           tumorUnits);

        if (is_empty(intersectionBuffer, dataTable.numTumor))
          continue;

        int TP = bitCollection_size(intersectionBuffer, dataTable.numTumor);

        // TODO: Use seperated intersectionBuffer
        const unit_t *rowIN = dataTable.normalData + computed.i * normalUnits;
        const unit_t *rowJN = dataTable.normalData + computed.j * normalUnits;
        const unit_t *rowKN = dataTable.normalData + k * normalUnits;
        const unit_t *rowLN = dataTable.normalData + l * normalUnits;
        for (size_t b = 0; b < normalUnits; b++) {
          intersectionBuffer[b] = rowIN[b] & rowJN[b] & rowKN[b] & rowLN[b];
        }

        int coveredNormal =
            bitCollection_size(intersectionBuffer, dataTable.numNormal);
        int TN = dataTable.numNormal - coveredNormal;
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

bool process_and_communicate(int rank, unit_t num_Comb, double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb, unit_t &begin,
                             unit_t &end, MPI_Status &status, sets_t dataTable,
                             unit_t *&intersectionBuffer,
                             unit_t *&scratchBufferij,
                             unit_t *&scratchBufferijk) {

  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          intersectionBuffer, scratchBufferij,
                          scratchBufferijk);
  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

  unit_t next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);

  if (next_idx == -1)
    return false;

  begin = next_idx;
  end = std::min(next_idx + CHUNK_SIZE, num_Comb);
  return true;
}

void worker_process(int rank, unit_t num_Comb, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb, sets_t dataTable,
                    unit_t *&intersectionBuffer, unit_t *&scratchBufferij,
                    unit_t *&scratchBufferijk) {

  std::pair<unit_t, unit_t> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);
  unit_t begin = chunk_indices.first;
  unit_t end = chunk_indices.second;
  MPI_Status status;
  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, localBestMaxF, localComb, begin, end, status, dataTable,
        intersectionBuffer, scratchBufferij, scratchBufferijk);
    if (!has_next)
      break;
  }
}

void execute_role(int rank, int size_minus_one, unit_t num_Comb,
                  double &localBestMaxF, std::array<int, NUMHITS> &localComb,
                  sets_t dataTable, unit_t *&intersectionBuffer,
                  unit_t *&scratchBufferij, unit_t *&scratchBufferijk) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, localBestMaxF, localComb, dataTable,
                   intersectionBuffer, scratchBufferij, scratchBufferijk);
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

bool all_bits_set(const unit_t *droppedSamples, size_t units) {
  return std::all_of(droppedSamples, droppedSamples + units,
                     [](unit_t val) { return val == ~static_cast<unit_t>(0); });
}

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable) {

  int Nt = dataTable.numTumor;
  int Nn = dataTable.numNormal;
  int numGenes = dataTable.numRows;

  size_t tumorUnits = UNITS_FOR_BITS(Nt);
  size_t normalUnits = UNITS_FOR_BITS(Nn);
  size_t maxUnits = std::max(tumorUnits, normalUnits);

  unit_t *intersectionBuffer = new unit_t[maxUnits];
  unit_t *scratchBufferij = new unit_t[normalUnits];
  unit_t *scratchBufferijk = new unit_t[tumorUnits];

  MPI_Datatype MPI_RESULT_WITH_COMB = create_mpi_result_with_comb_type();
  MPI_Op MPI_MAX_F_WITH_COMB = create_max_f_with_comb_op(MPI_RESULT_WITH_COMB);
  unit_t num_Comb = nCr(numGenes, 2);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  unit_t *droppedSamples = initialize_dropped_samples(CALCULATE_BIT_UNITS(Nt));

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (!all_bits_set(droppedSamples, CALCULATE_BIT_UNITS(Nt))) {
    double localBestMaxF;
    std::array<int, NUMHITS> localComb =
        initialize_local_comb_and_f(localBestMaxF);

    START_TIMING(master_worker)
    execute_role(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable,
                 intersectionBuffer, scratchBufferij, scratchBufferijk);
    END_TIMING(master_worker, master_worker_time);

    START_TIMING(all_reduce)
    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult;
#ifdef ALL_REDUCE_HIERARCHICAL
    Allreduce_hierarchical(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                  MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);
#else
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                  MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);
#endif  ///ALL_REDUCE_HIERARCHICAL

    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);
    END_TIMING(all_reduce, all_reduce_time);

    load_first_tumor(intersectionBuffer, dataTable, globalBestComb[0]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[1]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[2]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[3]);

    update_dropped_samples(droppedSamples, intersectionBuffer, tumorUnits);

    update_tumor_data(dataTable.tumorData, intersectionBuffer, tumorUnits,
                      numGenes);

    Nt -= bitCollection_size(intersectionBuffer, dataTable.numTumor);

    if (rank == 0) {
      write_output(rank, outfile, globalBestComb, globalResult.f);
    }
    // break;
  }
  if (rank == 0) {
    outfile.close();
  }

  elapsed_times[MASTER_WORKER] = master_worker_time;
  elapsed_times[ALL_REDUCE] = all_reduce_time;
  elapsed_times[BCAST] = broadcast_time;

  delete[] droppedSamples;
  MPI_Op_free(&MPI_MAX_F_WITH_COMB);
  MPI_Type_free(&MPI_RESULT_WITH_COMB);
}
