#include "commons.h"

#ifndef CARCUTILS_H
#define CARCUTILS_H

#define MAX_BUF_SIZE 1024
enum time_stages {
  MASTER_WORKER,
  ALL_REDUCE,
  BCAST,
  OVERALL_FILE_LOAD,
  OVERALL_DISTRIBUTE_FUNCTION,
  OVERALL_TOTAL
};

struct LambdaComputed {
  int i, j;
};

// ############MACROS####################
#ifdef ENABLE_TIMING
#define START_TIMING(var) double var##_start = MPI_Wtime();
#define END_TIMING(var, accumulated_time)                                      \
  do {                                                                         \
    double var##_end = MPI_Wtime();                                            \
    accumulated_time += var##_end - var##_start;                               \
  } while (0)
#else
#define START_TIMING(var)
#define END_TIMING(var, accumulated_time)
#endif

inline void load_first_tumor(unit_t *scratch, sets_t &table, size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;

  std::memcpy(scratch, &table.tumorData[baseIdx], rowUnits * sizeof(unit_t));
}

inline void load_first_normal(unit_t *scratch, sets_t &table, size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  std::memcpy(scratch, &table.normalData[baseIdx], rowUnits * sizeof(unit_t));
}

inline void inplace_intersect_tumor(unit_t *scratch, sets_t &table,
                                    size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;

  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] &= table.tumorData[baseIdx + b];
  }
}

inline void inplace_intersect_normal(unit_t *scratch, sets_t &table,
                                     size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] &= table.normalData[baseIdx + b];
  }
}

#endif
// ############MACROS####################
