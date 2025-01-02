#ifndef FOURHIT_H
#define FOURHIT_H

#include <array>
#include <mpi.h>
#include <string>

#include "constants.h"
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

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable);
#endif
