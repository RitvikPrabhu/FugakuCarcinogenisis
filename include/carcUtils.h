#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

typedef uint64_t unit_t;
#define MPI_UNIT_T MPI_UINT64_T

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  unit_t *tumorData;
  unit_t *normalData;
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

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)

#define ALL_BITS_SET (~(unit_t)0)

#define UNITS_FOR_BITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

inline bool is_empty(const uint64_t *buf, size_t units) {
  for (size_t i = 0; i < units; i++) {
    if (buf[i] != 0ULL)
      return false;
  }
  return true;
}

inline int bitCollection_size(const uint64_t *buf, size_t units) {
  int count = 0;
  for (size_t i = 0; i < units; i++) {
    count += __builtin_popcountll(buf[i]);
  }
  return count;
}

inline void load_first_tumor(unit_t *scratch, const sets_t &table,
                             size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;

  std::memcpy(scratch, &table.tumorData[baseIdx], rowUnits * sizeof(unit_t));
}

inline void inplace_intersect_tumor(unit_t *scratch, const sets_t &table,
                                    size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numTumor);
  size_t baseIdx = gene * rowUnits;

  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] &= table.tumorData[baseIdx + b];
  }
}

inline void load_first_normal(unit_t *scratch, const sets_t &table,
                              size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  for (size_t b = 0; b < rowUnits; b++) {
    scratch[b] = table.normalData[baseIdx + b];
  }
}

inline void inplace_intersect_normal(unit_t *scratch, const sets_t &table,
                                     size_t gene) {
  size_t rowUnits = UNITS_FOR_BITS(table.numNormal);
  size_t baseIdx = gene * rowUnits;
  std::memcpy(scratch, &table.normalData[baseIdx], rowUnits * sizeof(unit_t));
}

#endif
// ############MACROS####################
