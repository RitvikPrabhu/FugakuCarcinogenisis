#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
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
// #ifdef ENABLE_TIMING
// #endif

// ############HELPER FUNCTIONS####################
struct LambdaComputed {
  int i, j;
};

struct MPIResult {
  double value;
  int rank;
};

LambdaComputed compute_lambda_variables(long long int lambda) {
  LambdaComputed computed;
  computed.j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
  computed.i = lambda - (computed.j * (computed.j - 1)) / 2;
  return computed;
}

std::set<int> get_intersection(const std::set<int> &set1,
                               const std::set<int> &set2) {
  std::set<int> result;
  std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                        std::inserter(result, result.begin()));
  return result;
}

std::set<int>
compute_intersection_four_sets(const std::vector<std::set<int>> &data, int i,
                               int j, int k, int l) {
  std::set<int> intersect1 = get_intersection(data[i], data[j]);
  std::set<int> intersect2 = get_intersection(intersect1, data[k]);
  std::set<int> sampleToCover = get_intersection(intersect2, data[l]);
  return sampleToCover;
}

bool is_empty(const std::set<int> &set) { return set.empty(); }

double compute_F(int TP, int TN, double alpha) { return alpha * TP + TN; }

void update_best_combination(double &globalMaxF,
                             std::array<int, 4> &globalBestCombination,
                             double localMaxF,
                             const std::array<int, 4> &localBestCombination) {
  if (localMaxF >= globalMaxF) {
    globalMaxF = localMaxF;
    globalBestCombination = localBestCombination;
  }
}

void execute_role(int rank, int size_minus_one, long long int num_Comb,
                  std::vector<std::set<int>> &tumorData,
                  const std::vector<std::set<int>> &normalData, int numGenes,
                  int Nt, int Nn, double &localBestMaxF,
                  std::array<int, 4> &localComb) {
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

void perform_MPI_bcast(std::array<int, 4> &globalBestComb, int root_rank) {
  MPI_Bcast(globalBestComb.data(), 4, MPI_INT, root_rank, MPI_COMM_WORLD);
}

void update_tumor_data(std::vector<std::set<int>> &tumorData,
                       const std::set<int> &sampleToCover) {
  for (auto &tumorSet : tumorData) {
    for (const int sample : sampleToCover) {
      tumorSet.erase(sample);
    }
  }
}

void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void write_output(int rank, std::ofstream &outfile,
                  const std::array<int, 4> &globalBestComb,
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

bool process_and_communicate(
    int rank, long long int num_Comb, std::vector<std::set<int>> &tumorData,
    const std::vector<std::set<int>> &normalData, int numGenes, int Nt, int Nn,
    double &localBestMaxF, std::array<int, 4> &localComb, long long int &begin,
    long long int &end, MPI_Status &status) {
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

// ############MAIN FUNCTIONS####################

void process_lambda_interval(const std::vector<std::set<int>> &tumorData,
                             const std::vector<std::set<int>> &normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes,
                             std::array<int, 4> &bestCombination, int Nt,
                             int Nn, double &maxF) {
  const double alpha = 0.1;

#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, 4> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (long long int lambda = startComb; lambda <= endComb; lambda++) {
      LambdaComputed computed = compute_lambda_variables(lambda);

      const std::set<int> &gene1Tumor = tumorData[computed.i];
      const std::set<int> &gene2Tumor = tumorData[computed.j];
      std::set<int> intersectTumor1 = get_intersection(gene1Tumor, gene2Tumor);

      if (is_empty(intersectTumor1))
        continue;

      for (int k = computed.j + 1; k < totalGenes; k++) {
        const std::set<int> &gene3Tumor = tumorData[k];
        std::set<int> intersectTumor2 =
            get_intersection(gene3Tumor, intersectTumor1);

        if (is_empty(intersectTumor2))
          continue;

        for (int l = k + 1; l < totalGenes; l++) {
          const std::set<int> &gene4Tumor = tumorData[l];
          std::set<int> intersectTumor3 =
              get_intersection(gene4Tumor, intersectTumor2);

          if (is_empty(intersectTumor3))
            continue;

          std::set<int> intersectNormal = compute_intersection_four_sets(
              normalData, computed.i, computed.j, k, l);

          int TP = static_cast<int>(intersectTumor3.size());
          int TN = static_cast<int>(Nn - intersectNormal.size());

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
                    std::vector<std::set<int>> &tumorData,
                    const std::vector<std::set<int>> &normalData, int numGenes,
                    int Nt, int Nn, double &localBestMaxF,
                    std::array<int, 4> &localComb) {

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
                      std::vector<std::set<int>> &tumorData,
                      std::vector<std::set<int>> &normalData, int Nt, int Nn,
                      const char *outFilename,
                      const std::set<int> &tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]) {

  long long int num_Comb = nCr(numGenes, 2);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  std::set<int> droppedSamples;

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (tumorSamples != droppedSamples) {
    std::array<int, 4> localComb = {-1, -1, -1, -1};
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

    std::array<int, 4> globalBestComb = {-1, -1, -1, -1};
    if (rank == globalResult.rank) {
      globalBestComb = localComb;
    }

    START_TIMING(broadcast)
    perform_MPI_bcast(globalBestComb, globalResult.rank);
    END_TIMING(broadcast, broadcast_time);

    std::set<int> sampleToCover = compute_intersection_four_sets(
        tumorData, globalBestComb[0], globalBestComb[1], globalBestComb[2],
        globalBestComb[3]);
    droppedSamples.insert(sampleToCover.begin(), sampleToCover.end());

    update_tumor_data(tumorData, sampleToCover);
    if (rank == 0) {
      write_output(rank, outfile, globalBestComb, geneIdArray,
                   globalResult.value);
    }
    break;
  }
  if (rank == 0) {
    outfile.close();
  }

  elapsed_times[MASTER_WORKER] = master_worker_time;
  elapsed_times[ALL_REDUCE] = all_reduce_time;
  elapsed_times[BCAST] = broadcast_time;
}
