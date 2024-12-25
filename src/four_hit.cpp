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
  double A;
  unsigned long long int k_long;
  unsigned long long int Tz;
  int i, j, k;
};

struct MPIResult {
  double value;
  int rank;
};

LambdaComputed compute_lambda_variables(long long int lambda) {
  LambdaComputed computed;
  double term1 = 243.0 * lambda - 1.0 / lambda;
  double rhs = (std::log(3.0 * lambda) + std::log(term1)) / 2.0;
  computed.A = std::exp(rhs);

  double common_numerator = std::pow(computed.A + 27.0 * lambda, 1.0 / 3.0);
  double common_denominator = std::pow(3.0, 2.0 / 3.0);
  double v = (common_numerator / common_denominator) +
             (1.0 / (common_numerator * std::pow(3.0, 1.0 / 3.0))) - 1.0;

  computed.k_long = static_cast<unsigned long long int>(v);
  computed.Tz =
      computed.k_long * (computed.k_long + 1) * (computed.k_long + 2) / 6;
  long long int LambdaP = lambda - computed.Tz;

  computed.k = static_cast<int>(computed.k_long);
  computed.j = static_cast<int>(std::sqrt(0.25 + 2.0 * LambdaP) - 0.5);
  unsigned long long int T2Dy = computed.j * (computed.j + 1) / 2;
  computed.i = static_cast<int>(LambdaP - T2Dy);

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
                  long long int &count, int Nt, int Nn, double &localBestMaxF,
                  std::array<int, 4> &localComb) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, tumorData, normalData, numGenes, count, Nt,
                   Nn, localBestMaxF, localComb);
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

void write_output(int rank, const char *outFilename,
                  const std::array<int, 4> &globalBestComb,
                  const std::string *geneIdArray, double F_max) {
  if (rank == 0) {
    std::ofstream outfile(outFilename, std::ios::app);
    if (!outfile) {
      std::cerr << "Error: Could not open output file." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    outfile << "(";
    for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
      outfile << geneIdArray[globalBestComb[idx]];
      if (idx != globalBestComb.size() - 1) {
        outfile << ", ";
      }
    }
    outfile << ")  F-max = " << F_max << std::endl;
    outfile.close();
  }
}

// ############MAIN FUNCTIONS####################

void process_lambda_interval(const std::vector<std::set<int>> &tumorData,
                             const std::vector<std::set<int>> &normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes, long long int &count,
                             std::array<int, 4> &bestCombination, int Nt,
                             int Nn, double &maxF) {
  const double alpha = 0.1;

#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, 4> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (long long int lambda = startComb; lambda <= endComb; lambda++) {
      if (lambda <= 0)
        continue;

      LambdaComputed computed = compute_lambda_variables(lambda);

      if (computed.i >= computed.j || computed.j >= computed.k ||
          computed.i >= computed.k)
        continue;

      const std::set<int> &gene1Tumor = tumorData[computed.i];
      const std::set<int> &gene2Tumor = tumorData[computed.j];
      std::set<int> intersectTumor1 = get_intersection(gene1Tumor, gene2Tumor);

      if (is_empty(intersectTumor1))
        continue;

      const std::set<int> &gene3Tumor = tumorData[computed.k];
      std::set<int> intersectTumor2 =
          get_intersection(gene3Tumor, intersectTumor1);

      if (is_empty(intersectTumor2))
        continue;

      for (int l = computed.k + 1; l < totalGenes; l++) {
        const std::set<int> &gene4Tumor = tumorData[l];
        std::set<int> intersectTumor3 =
            get_intersection(gene4Tumor, intersectTumor2);

        if (is_empty(intersectTumor3))
          continue;

        std::set<int> intersectNormal = compute_intersection_four_sets(
            normalData, computed.i, computed.j, computed.k, l);

        int TP = static_cast<int>(intersectTumor3.size());
        int TN = static_cast<int>(Nn - intersectNormal.size());

        double F = compute_F(TP, TN, alpha);
        if (F >= localMaxF) {
          localMaxF = F;
          localBestCombination = {computed.i, computed.j, computed.k, l};
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
                    long long int &count, int Nt, int Nn, double &localBestMaxF,
                    std::array<int, 4> &localComb) {

  long long int begin = (rank - 1) * CHUNK_SIZE;
  long long int end = std::min(begin + CHUNK_SIZE, num_Comb);
  MPI_Status status;
  while (end <= num_Comb) {
    process_lambda_interval(tumorData, normalData, begin, end, numGenes, count,
                            localComb, Nt, Nn, localBestMaxF);
    char c = 'a';
    MPI_Send(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

    long long int next_idx;
    MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);
    if (next_idx == -1)
      break;

    begin = next_idx;
    end = std::min(next_idx + CHUNK_SIZE, num_Comb);
  }
}

void distribute_tasks(int rank, int size, int numGenes,
                      std::vector<std::set<int>> &tumorData,
                      std::vector<std::set<int>> &normalData,
                      long long int &count, int Nt, int Nn,
                      const char *outFilename,
                      const std::set<int> &tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]) {

  long long int num_Comb = nCr(numGenes, 3);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  std::set<int> droppedSamples;

  while (tumorSamples != droppedSamples) {
    std::array<int, 4> localComb = {-1, -1, -1, -1};
    double localBestMaxF = -1.0;

    START_TIMING(master_worker)
    execute_role(rank, size - 1, num_Comb, tumorData, normalData, numGenes,
                 count, Nt, Nn, localBestMaxF, localComb);
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

    write_output(rank, outFilename, globalBestComb, geneIdArray,
                 globalResult.value);
  }

  elapsed_times[0] = master_worker_time;
  elapsed_times[1] = all_reduce_time;
  elapsed_times[2] = broadcast_time;
}
