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

struct LambdaComputed {
  double A;
  unsigned long long int k_long;
  unsigned long long int Tz;
  int i, j, k;
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

        const std::set<int> &gene1Normal = normalData[computed.i];
        const std::set<int> &gene2Normal = normalData[computed.j];
        const std::set<int> &gene3Normal = normalData[computed.k];
        const std::set<int> &gene4Normal = normalData[l];

        std::set<int> intersectNormal1 =
            get_intersection(gene1Normal, gene2Normal);
        std::set<int> intersectNormal2 =
            get_intersection(gene3Normal, intersectNormal1);
        std::set<int> intersectNormal3 =
            get_intersection(gene4Normal, intersectNormal2);

        int TP = static_cast<int>(intersectTumor3.size());
        int TN = static_cast<int>(Nn - intersectNormal3.size());

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
                    long long int &count, int Nt, int Nn, const char *hit3_file,
                    double &localBestMaxF, std::array<int, 4> &localComb) {

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
                      const char *outFilename, const char *hit3_file,
                      const std::set<int> &tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]) {

  long long int num_Comb = nCr(numGenes, 3);
  double start_time, end_time;
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  std::set<int> droppedSamples;
  while (tumorSamples != droppedSamples) {
    std::array<int, 4> localComb = {-1, -1, -1, -1};
    double localBestMaxF = -1.0;
    start_time = MPI_Wtime();
    if (rank == 0) { // Master
      master_process(size - 1, num_Comb);
    } else { // Worker
      worker_process(rank, num_Comb, tumorData, normalData, numGenes, count, Nt,
                     Nn, hit3_file, localBestMaxF, localComb);
    }
    end_time = MPI_Wtime();
    master_worker_time += end_time - start_time;
    struct {
      double value;
      int rank;
    } localResult, globalResult;

    localResult.value = localBestMaxF;
    localResult.rank = rank;

    start_time = MPI_Wtime();
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    all_reduce_time += end_time - start_time;

    std::array<int, 4> globalBestComb;
    if (rank == globalResult.rank) {
      globalBestComb = localComb;
    }

    start_time = MPI_Wtime();
    MPI_Bcast(globalBestComb.data(), 4, MPI_INT, globalResult.rank,
              MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    broadcast_time += end_time - start_time;

    std::set<int> finalIntersect1;
    std::set<int> finalIntersect2;
    std::set<int> sampleToCover;
    std::set_intersection(
        tumorData[globalBestComb[0]].begin(),
        tumorData[globalBestComb[0]].end(),
        tumorData[globalBestComb[1]].begin(),
        tumorData[globalBestComb[1]].end(),
        std::inserter(finalIntersect1, finalIntersect1.begin()));
    std::set_intersection(
        finalIntersect1.begin(), finalIntersect1.end(),
        tumorData[globalBestComb[2]].begin(),
        tumorData[globalBestComb[2]].end(),
        std::inserter(finalIntersect2, finalIntersect2.begin()));
    std::set_intersection(finalIntersect2.begin(), finalIntersect2.end(),
                          tumorData[globalBestComb[3]].begin(),
                          tumorData[globalBestComb[3]].end(),
                          std::inserter(sampleToCover, sampleToCover.begin()));

    droppedSamples.insert(sampleToCover.begin(), sampleToCover.end());

    for (auto &tumorSet : tumorData) {
      for (const int sample : sampleToCover) {
        tumorSet.erase(sample);
      }
    }

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
      outfile << ")  F-max = " << globalResult.value << std::endl;
      outfile.close();
    }
  }

  elapsed_times[MASTER_WORKER] = master_worker_time;
  elapsed_times[ALL_REDUCE] = all_reduce_time;
  elapsed_times[BCAST] = broadcast_time;
}
