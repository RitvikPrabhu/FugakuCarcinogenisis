#ifndef FOURHIT_H
#define FOURHIT_H

#include <array>
#include <mpi.h>
#include <string>

#include "commons.h"
#include "utils.h"

struct MPIResultWithComb {
  double f;
  int comb[NUMHITS];
};

struct MPIResult {
  double value;
  int rank;
};

struct LambdaComputed {
  int i, j;
};

using LAMBDA_TYPE = long long;

#ifdef USE_CPP_SET
#define SET_INTERSECT_N(dest, size_in_units, ...)                              \
  do {                                                                         \
    SET temp_sets[] = {__VA_ARGS__};                                           \
    SET_INTERSECT(dest, temp_sets[0], temp_sets[1], size_in_units);            \
    for (size_t idx = 2; idx < sizeof(temp_sets) / sizeof(temp_sets[0]);       \
         ++idx) {                                                              \
      SET_INTERSECT(dest, dest, temp_sets[idx], size_in_units);                \
    }                                                                          \
  } while (0)
#else
#define SET_INTERSECT_N(dest, size_in_units, ...)                              \
  do {                                                                         \
    SET temp_sets[] = {__VA_ARGS__};                                           \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      dest[__i] = temp_sets[0][__i];                                           \
      for (size_t idx = 1; idx < sizeof(temp_sets) / sizeof(temp_sets[0]);     \
           ++idx)                                                              \
        dest[__i] &= temp_sets[idx][__i];                                      \
    }                                                                          \
  } while (0)
#endif

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable);
#endif
