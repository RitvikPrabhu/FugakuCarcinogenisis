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

unit_t nCr(int n, int r) {
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

unit_t *allocate_bit_array(size_t units,
                           unit_t init_value = 0xFFFFFFFFFFFFFFFFULL) {
  unit_t *bitArray = new unit_t[units];
  for (size_t i = 0; i < units; ++i) {
    bitArray[i] = init_value;
  }
  return bitArray;
}

void bitwise_and_arrays(unit_t *result, const unit_t *source, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    result[i] &= source[i];
  }
}

unit_t *get_intersection(unit_t **data, int numSamples, ...) {
  size_t units = CALCULATE_BIT_UNITS(numSamples);
  unit_t *finalIntersect = allocate_bit_array(units);

  va_list args;
  va_start(args, numSamples);
  bool isFirst = true;
  while (true) {
    int geneIndex = va_arg(args, int);
    if (geneIndex == -1) {
      break;
    }

    if (isFirst) {
      for (size_t i = 0; i < units; ++i) {
        finalIntersect[i] = data[geneIndex][i];
      }
      isFirst = false;
    } else {
      bitwise_and_arrays(finalIntersect, data[geneIndex], units);
    }
  }
  va_end(args);
  return finalIntersect; // Return single pointer
}

bool is_empty(unit_t *bitArray, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    if (bitArray[i] != 0) {
      return false;
    }
  }
  return true;
}

size_t bitCollection_size(unit_t *bitArray, size_t units) {
  size_t count = 0;
  for (size_t i = 0; i < units; ++i) {
    count += __builtin_popcountll(bitArray[i]);
  }
  return count;
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

void update_tumor_data(unit_t **&tumorData, unit_t *sampleToCover, size_t units,
                       int numGenes) {
  for (int gene = 0; gene < numGenes; ++gene) {
    for (size_t i = 0; i < units; ++i) {
      tumorData[gene][i] &= ~sampleToCover[i];
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

void updateNt(int &Nt, unit_t *&sampleToCover) {
  Nt -= bitCollection_size(sampleToCover, CALCULATE_BIT_UNITS(Nt));
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

void update_best_combination(
    double &globalMaxF, std::array<int, NUMHITS> &globalBestCombination,
    double localMaxF, const std::array<int, NUMHITS> &localBestCombination) {
  if (localMaxF >= globalMaxF) {
    globalMaxF = localMaxF;
    globalBestCombination = localBestCombination;
  }
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
                             double &maxF, sets_t dataTable) {
  const double alpha = 0.1;

#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, NUMHITS> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (unit_t lambda = startComb; lambda <= endComb; lambda++) {
      LambdaComputed computed =
          compute_lambda_variables(lambda, dataTable.numRows);
      if (computed.j == -1)
        continue;

      unit_t *intersectTumor1 = get_intersection(tumorData, dataTable.numTumor,
                                                 computed.i, computed.j, -1);

      if (is_empty(intersectTumor1, CALCULATE_BIT_UNITS(dataTable.numTumor)))
        continue;

      for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {
        unit_t *intersectTumor2 = get_intersection(tumorData, dataTable.numTumor), computed.i, computed.j, k, -1);

        if (is_empty(intersectTumor2, CALCULATE_BIT_UNITS(dataTable.numTumor)))
          continue;

        for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
          unit_t *intersectTumor3 = get_intersection(
              tumorData, dataTable.numTumor, computed.i, computed.j, k, l, -1);

          if (is_empty(intersectTumor3,
                       CALCULATE_BIT_UNITS(dataTable.numTumor)))
            continue;

          unit_t *intersectNormal =
              get_intersection(normalData, dataTable.numNormal, computed.i,
                               computed.j, k, l, -1);

          int TP = bitCollection_size(intersectTumor3,
                                      CALCULATE_BIT_UNITS(dataTable.numTumor));
          int TN = dataTable.numNormal -
                   bitCollection_size(intersectNormal,
                                      CALCULATE_BIT_UNITS(dataTable.numNormal));

          double F =
              compute_F(TP, TN, alpha, dataTable.numTumor, dataTable.numNormal);
          if (F >= localMaxF) {
            localMaxF = F;
            localBestCombination = {computed.i, computed.j, k, l};
          }
        }
      }
    }

#pragma omp critical
    {
      update_best_combination(maxF, bestCombination, localMaxF,
                              localBestCombination);
    }
  }
}

bool process_and_communicate(int rank, unit_t num_Comb, double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb, unit_t &begin,
                             unit_t &end, MPI_Status &status,
                             sets_t dataTable) {

  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable);
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
                    std::array<int, NUMHITS> &localComb, sets_t dataTable) {

  std::pair<unit_t, unit_t> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);
  unit_t begin = chunk_indices.first;
  unit_t end = chunk_indices.second;
  MPI_Status status;
  while (end <= num_Comb) {
    bool has_next =
        process_and_communicate(rank, num_Comb, localBestMaxF, localComb, begin,
                                end, status, dataTable);
    if (!has_next)
      break;
  }
}

void execute_role(int rank, int size_minus_one, unit_t num_Comb,
                  double &localBestMaxF, std::array<int, NUMHITS> &localComb,
                  sets_t dataTable) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, localBestMaxF, localComb, dataTable);
  }
}

std::array<int, NUMHITS> initialize_local_comb_and_f(double &f) {
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
    execute_role(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable);
    END_TIMING(master_worker, master_worker_time);

    START_TIMING(all_reduce)
    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult;
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                  MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);

    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);
    END_TIMING(all_reduce, all_reduce_time);

    unit_t *sampleToCover =
        get_intersection(tumorData, Nt, globalBestComb[0], globalBestComb[1],
                         globalBestComb[2], globalBestComb[3], -1);
    update_dropped_samples(droppedSamples, sampleToCover,
                           CALCULATE_BIT_UNITS(Nt));

    update_tumor_data(tumorData, sampleToCover, CALCULATE_BIT_UNITS(Nt),
                      numGenes);

    updateNt(Nt, sampleToCover);
    delete[] sampleToCover;
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
