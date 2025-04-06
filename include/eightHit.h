#ifndef EIGHTHIT_H
#define EIGHTHIT_H

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

#define SET_INTERSECT8(dest, A, B, C, D, E, F, G, H, size_in_units)            \
  do {                                                                         \
    SET_INTERSECT(dest, A, B, size_in_units);                                  \
    SET_INTERSECT(dest, dest, C, size_in_units);                               \
    SET_INTERSECT(dest, dest, D, size_in_units);                               \
    SET_INTERSECT(dest, dest, E, size_in_units);                               \
    SET_INTERSECT(dest, dest, F, size_in_units);                               \
    SET_INTERSECT(dest, dest, G, size_in_units);                               \
    SET_INTERSECT(dest, dest, H, size_in_units);                               \
  } while (0)

#else
#define SET_INTERSECT8(dest, A, B, C, D, E, F, G, H, size_in_units)            \
  do {                                                                         \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      (dest)[__i] = (A)[__i] & (B)[__i] & (C)[__i] & (D)[__i] & (E)[__i] &     \
                    (F)[__i] & (G)[__i] & (H)[__i];                            \
    }                                                                          \
  } while (0)

#endif

void distribute_tasks(int rank, int size, const char *outFilename,
                      const char *csvFileName, sets_t dataTable,
                      double *omit_time);
#endif
