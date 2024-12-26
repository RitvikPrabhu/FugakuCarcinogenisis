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

#include "constants.h"
#include "four_hit.h"
#include "mpi_specific.h"
#include "utils.h"

#define NUMHITS 4
// ############HELPER FUNCTIONS####################
struct LambdaComputed {
  int i, j;
};

struct MPIResult {
  double value;
  int rank;
};

LambdaComputed compute_lambda_variables(long long int lambda, int totalGenes) {
  LambdaComputed computed;
  computed.j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
  if (computed.j > totalGenes - (NUMHITS - 2)) {
    computed.j = -1;
    return computed;
  }
  computed.i = lambda - (computed.j * (computed.j - 1)) / 2;
  return computed;
}

unsigned long long *
allocate_bit_array(size_t units,
                   unsigned long long init_value = 0xFFFFFFFFFFFFFFFFULL) {
  unsigned long long *bitArray = new unsigned long long[units];
  for (size_t i = 0; i < units; ++i) {
    bitArray[i] = init_value;
  }
  return bitArray;
}

void bitwise_and_arrays(unsigned long long *result,
                        const unsigned long long *source, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    result[i] &= source[i];
  }
}

unsigned long long **get_intersection(unsigned long long **data, int numSamples,
                                      ...) {
  size_t units = calculate_bit_units(numSamples);
  unsigned long long **finalIntersect = new unsigned long long *[1];
  finalIntersect[0] = allocate_bit_array(units);

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
        finalIntersect[0][i] = data[geneIndex][i];
      }
      isFirst = false;
    } else {
      bitwise_and_arrays(finalIntersect[0], data[geneIndex], units);
    }
  }

  va_end(args);
  return finalIntersect;
}

bool is_empty(unsigned long long **bitArray, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    if (bitArray[0][i] != 0) {
      return false;
    }
  }
  return true;
}

size_t bitCollection_size(unsigned long long **bitArray, size_t units) {
  size_t count = 0;
  for (size_t i = 0; i < units; ++i) {
    count += __builtin_popcountll(bitArray[0][i]);
  }
  return count;
}

double compute_F(int TP, int TN, double alpha) { return alpha * TP + TN; }

void update_best_combination(
    double &globalMaxF, std::array<int, NUMHITS> &globalBestCombination,
    double localMaxF, const std::array<int, NUMHITS> &localBestCombination) {
  if (localMaxF >= globalMaxF) {
    globalMaxF = localMaxF;
    globalBestCombination = localBestCombination;
  }
}

void execute_role(int rank, int size_minus_one, long long int num_Comb,
                  unsigned long long **&tumorData,
                  unsigned long long **&normalData, int numGenes, int Nt,
                  int Nn, double &localBestMaxF,
                  std::array<int, NUMHITS> &localComb) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, tumorData, normalData, numGenes, Nt, Nn,
                   localBestMaxF, localComb);
  }
}

MPIResult perform_MPI_allreduce(const MPIResult &localResult) {
  MPIResult globalResult;
  MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                MPI_COMM_WORLD);
  return globalResult;
}

void perform_MPI_bcast(std::array<int, NUMHITS> &globalBestComb,
                       int root_rank) {
  MPI_Bcast(globalBestComb.data(), NUMHITS, MPI_INT, root_rank, MPI_COMM_WORLD);
}

void update_tumor_data(unsigned long long **&tumorData,
                       unsigned long long **sampleToCover, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    tumorData[0][i] &= ~sampleToCover[0][i];
  }
}

void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void write_output(int rank, std::ofstream &outfile,
                  const std::array<int, NUMHITS> &globalBestComb,
                  const std::string *geneIdArray, double F_max) {
  outfile << "(";
  for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
    outfile << geneIdArray[globalBestComb[idx]];
    if (idx != globalBestComb.size() - 1) {
      outfile << ", ";
    }
  }
  outfile << ")  F-max = " << F_max << std::endl;
}

void notify_master_chunk_processed(int master_rank = 0, int tag = 1) {
  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, master_rank, tag, MPI_COMM_WORLD);
}

long long int receive_next_chunk_index(MPI_Status &status, int master_rank = 0,
                                       int tag = 2) {
  long long int next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, master_rank, tag, MPI_COMM_WORLD,
           &status);
  return next_idx;
}

std::pair<long long int, long long int>
calculate_initial_chunk(int rank, long long int num_Comb,
                        long long int chunk_size) {
  long long int begin = (rank - 1) * chunk_size;
  long long int end = std::min(begin + chunk_size, num_Comb);
  return {begin, end};
}

bool process_and_communicate(int rank, long long int num_Comb,
                             unsigned long long **&tumorData,
                             unsigned long long **&normalData, int numGenes,
                             int Nt, int Nn, double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb,
                             long long int &begin, long long int &end,
                             MPI_Status &status) {
  // Process the current chunk
  process_lambda_interval(tumorData, normalData, begin, end, numGenes,
                          localComb, Nt, Nn, localBestMaxF);

  // Notify the master that the chunk has been processed
  notify_master_chunk_processed();

  // Receive the next chunk index from the master
  long long int next_idx = receive_next_chunk_index(status);
  // Check if there are more chunks to process
  if (next_idx == -1)
    return false;

  // Update the begin and end indices for the next chunk
  begin = next_idx;
  end = std::min(next_idx + CHUNK_SIZE, num_Comb);
  return true;
}

void update_dropped_samples(unsigned long long *&droppedSamples,
                            unsigned long long **sampleToCover, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    droppedSamples[0] |= sampleToCover[0][i];
  }
}

unsigned long long *initialize_dropped_samples(size_t units) {
  unsigned long long *droppedSamples = new unsigned long long[units];
  memset(droppedSamples, 0, units * sizeof(unsigned long long));
  return droppedSamples;
}

bool arrays_equal(const unsigned long long *a, const unsigned long long *b,
                  size_t units) {
  for (size_t i = 0; i < units; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// ############MAIN FUNCTIONS####################

void process_lambda_interval(unsigned long long **&tumorData,
                             unsigned long long **&normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes,
                             std::array<int, NUMHITS> &bestCombination, int Nt,
                             int Nn, double &maxF) {
  const double alpha = 0.1;

#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, NUMHITS> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (long long int lambda = startComb; lambda <= endComb; lambda++) {
      LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
      if (computed.j == -1)
        continue;

      unsigned long long **intersectTumor1 =
          get_intersection(tumorData, Nt, computed.i, computed.j, -1);

      if (is_empty(intersectTumor1, calculate_bit_units(Nt)))
        continue;

      for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {
        unsigned long long **intersectTumor2 =
            get_intersection(tumorData, Nt, computed.i, computed.j, k, -1);

        if (is_empty(intersectTumor2, calculate_bit_units(Nt)))
          continue;

        for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
          unsigned long long **intersectTumor3 =
              get_intersection(tumorData, Nt, computed.i, computed.j, k, l, -1);

          if (is_empty(intersectTumor3, calculate_bit_units(Nt)))
            continue;

          unsigned long long **intersectNormal =
              get_intersection(tumorData, Nn, computed.i, computed.j, k, l, -1);

          int TP = bitCollection_size(intersectTumor3, calculate_bit_units(Nt));
          int TN =
              Nn - bitCollection_size(intersectTumor3, calculate_bit_units(Nn));

          double F = compute_F(TP, TN, alpha);
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

void worker_process(int rank, long long int num_Comb,
                    unsigned long long **&tumorData,
                    unsigned long long **&normalData, int numGenes, int Nt,
                    int Nn, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb) {

  std::pair<long long int, long long int> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);
  long long int begin = chunk_indices.first;
  long long int end = chunk_indices.second;
  MPI_Status status;
  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, tumorData, normalData, numGenes, Nt, Nn, localBestMaxF,
        localComb, begin, end, status);
    if (!has_next)
      break;
  }
}

void distribute_tasks(int rank, int size, int numGenes,
                      unsigned long long **&tumorData,
                      unsigned long long **&normalData, int Nt, int Nn,
                      const char *outFilename,
                      unsigned long long *&tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]) {

  long long int num_Comb = nCr(numGenes, 2);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  unsigned long long *droppedSamples =
      initialize_dropped_samples(calculate_bit_units(Nt));

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (!arrays_equal(tumorSamples, droppedSamples, calculate_bit_units(Nt))) {
    std::array<int, NUMHITS> localComb = {-1, -1, -1, -1};
    double localBestMaxF = -1.0;

    START_TIMING(master_worker)
    execute_role(rank, size - 1, num_Comb, tumorData, normalData, numGenes, Nt,
                 Nn, localBestMaxF, localComb);
    END_TIMING(master_worker, master_worker_time);

    MPIResult localResult;
    localResult.value = localBestMaxF;
    localResult.rank = rank;

    START_TIMING(all_reduce)
    MPIResult globalResult = perform_MPI_allreduce(localResult);
    END_TIMING(all_reduce, all_reduce_time);

    std::array<int, NUMHITS> globalBestComb = {-1, -1, -1, -1};
    if (rank == globalResult.rank) {
      globalBestComb = localComb;
    }

    START_TIMING(broadcast)
    perform_MPI_bcast(globalBestComb, globalResult.rank);
    END_TIMING(broadcast, broadcast_time);

    unsigned long long **sampleToCover =
        get_intersection(tumorData, Nt, globalBestComb[0], globalBestComb[1],
                         globalBestComb[2], globalBestComb[3], -1);
    update_dropped_samples(droppedSamples, sampleToCover,
                           calculate_bit_units(Nt));

    update_tumor_data(tumorData, sampleToCover, calculate_bit_units(Nt));
    if (rank == 0) {
      write_output(rank, outfile, globalBestComb, geneIdArray,
                   globalResult.value); // TODO: This also needs to be updated
    }

    delete[] sampleToCover[0];
    delete[] sampleToCover;
  }
  if (rank == 0) {
    outfile.close();
  }

  elapsed_times[MASTER_WORKER] = master_worker_time;
  elapsed_times[ALL_REDUCE] = all_reduce_time;
  elapsed_times[BCAST] = broadcast_time;

  delete[] droppedSamples;
}
