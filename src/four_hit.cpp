#include "four_hit.h"
#include "constants.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <string>
#include <vector>

// #ifdef ENABLE_TIMING
// #endif

void process_lambda_interval(const std::vector<std::set<int>> &tumorData,
                             const std::vector<std::set<int>> &normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes, long long int &count,
                             std::array<int, 4> &bestCombination, int Nt,
                             int Nn, double &maxF) {
  double alpha = 0.1;
#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, 4> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (long long int lambda = startComb; lambda <= endComb; lambda++) {

      if (lambda <= 0)
        continue; // Avoid invalid lambda values

      double term1 = 243.0 * lambda - 1.0 / lambda;
      double rhs = (std::log(3.0 * lambda) + std::log(term1)) / 2.0;
      double A = std::exp(rhs);

      double common_numerator = std::pow(A + 27.0 * lambda, 1.0 / 3.0);
      double common_denominator = std::pow(3.0, 2.0 / 3.0);

      double v = (common_numerator / common_denominator) +
                 (1.0 / (common_numerator * std::pow(3.0, 1.0 / 3.0))) - 1.0;

      unsigned long long int k_long = static_cast<unsigned long long int>(v);
      unsigned long long int Tz = k_long * (k_long + 1) * (k_long + 2) / 6;

      unsigned long long int LambdaP = lambda - Tz;

      int k = static_cast<int>(k_long);
      int j = static_cast<int>(std::sqrt(0.25 + 2.0 * LambdaP) - 0.5);

      unsigned long long int T2Dy = j * (j + 1) / 2;

      int i = static_cast<int>(LambdaP - T2Dy);
      if (i >= j || j >= k || i >= k)
        continue;

      const std::set<int> &gene1Tumor = tumorData[i];
      const std::set<int> &gene2Tumor = tumorData[j];

      std::set<int> intersectTumor1;
      std::set_intersection(
          gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(),
          gene2Tumor.end(),
          std::inserter(intersectTumor1, intersectTumor1.begin()));

      if (!intersectTumor1.empty()) {
        const std::set<int> &gene3Tumor = tumorData[k];
        std::set<int> intersectTumor2;
        std::set_intersection(
            gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(),
            intersectTumor1.end(),
            std::inserter(intersectTumor2, intersectTumor2.begin()));

        if (!intersectTumor2.empty()) {
          for (int l = k + 1; l < totalGenes; l++) {
            const std::set<int> &gene4Tumor = tumorData[l];
            std::set<int> intersectTumor3;
            std::set_intersection(
                gene4Tumor.begin(), gene4Tumor.end(), intersectTumor2.begin(),
                intersectTumor2.end(),
                std::inserter(intersectTumor3, intersectTumor3.begin()));

            const std::set<int> &gene1Normal = normalData[i];
            const std::set<int> &gene2Normal = normalData[j];
            const std::set<int> &gene3Normal = normalData[k];
            const std::set<int> &gene4Normal = normalData[l];

            std::set<int> intersectNormal1;
            std::set<int> intersectNormal2;
            std::set<int> intersectNormal3;

            std::set_intersection(
                gene1Normal.begin(), gene1Normal.end(), gene2Normal.begin(),
                gene2Normal.end(),
                std::inserter(intersectNormal1, intersectNormal1.begin()));
            std::set_intersection(
                gene3Normal.begin(), gene3Normal.end(),
                intersectNormal1.begin(), intersectNormal1.end(),
                std::inserter(intersectNormal2, intersectNormal2.begin()));
            std::set_intersection(
                gene4Normal.begin(), gene4Normal.end(),
                intersectNormal2.begin(), intersectNormal2.end(),
                std::inserter(intersectNormal3, intersectNormal3.begin()));

            if (!intersectTumor3.empty()) {
              int TP = static_cast<int>(intersectTumor3.size());
              int TN = static_cast<int>(Nn - intersectNormal3.size());

              double F = static_cast<double>(alpha * TP + TN);
              if (F >= localMaxF) {
                localMaxF = F;
                localBestCombination = std::array<int, 4>{i, j, k, l};
              }
            }
          }
        }
      }
    }

#pragma omp critical
    {
      if (localMaxF >= maxF) {
        maxF = localMaxF;
        bestCombination = localBestCombination;
      }
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
