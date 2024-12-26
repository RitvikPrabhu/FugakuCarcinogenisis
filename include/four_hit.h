#ifndef FOUR_HIT_H
#define FOUR_HIT_H

#include <array>
#include <mpi.h>
#include <set>
#include <string>
#include <vector>

#define NUMHITS 4

struct MPIResultWithComb {
  double fscore;
  int iter[NUMHITS];
};

struct LambdaComputed {
  int i, j;
};

void process_lambda_interval(unsigned long long **&tumorData,
                             unsigned long long **&normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes,
                             std::array<int, 4> &bestCombination, int Nt,
                             int Nn, double &maxF);

void worker_process(int rank, long long int num_Comb,
                    unsigned long long **&tumorData,
                    unsigned long long **&normalData, int numGenes, int Nt,
                    int Nn, double &localBestMaxF,
                    std::array<int, 4> &localComb);

void distribute_tasks(int rank, int size, int numGenes,
                      unsigned long long **&tumorData,
                      unsigned long long **&normalData, int Nt, int Nn,
                      const char *outFilename,
                      unsigned long long *&tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]);

#endif
