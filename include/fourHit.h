#ifndef FOUR_HIT_H
#define FOUR_HIT_H

#include <array>
#include <mpi.h>
#include <string>

#define NUMHITS 4
#define CHUNK_SIZE 100LL

#define BITS_PER_UNIT 64
#define CALCULATE_BIT_UNITS(numSample)                                         \
  (((numSample) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

struct MPIResultWithComb {
  double f;
  int comb[NUMHITS];
};

struct LambdaComputed {
  int i, j;
};

struct MPIResult {
  double value;
  int rank;
};

void distribute_tasks(int rank, int size, int numGenes,
                      unsigned long long **&tumorData,
                      unsigned long long **&normalData, int Nt, int Nn,
                      const char *outFilename,
                      unsigned long long *&tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]);

#endif
