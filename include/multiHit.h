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
#define SET_INTERSECT_N(dest, sets_array, num_sets, size_in_units)             \
  do {                                                                         \
    SET_COPY(dest, sets_array[0], size_in_units);                              \
    for (size_t idx = 1; idx < num_sets; ++idx) {                              \
      SET_INTERSECT(dest, dest, sets_array[idx], size_in_units);               \
    }                                                                          \
  } while (0)
#else
#define SET_INTERSECT_N(dest, sets_array, num_sets, size_in_units)             \
  do {                                                                         \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      dest[__i] = sets_array[0][__i];                                          \
      for (size_t idx = 1; idx < num_sets; ++idx)                              \
        dest[__i] &= sets_array[idx][__i];                                     \
    }                                                                          \
  } while (0)
#endif

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable);
#endif
