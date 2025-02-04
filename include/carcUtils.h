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

inline bool is_empty(unit_t *buf, size_t validBits) {
  size_t fullUnits = validBits / BITS_PER_UNIT;
  size_t remainder = validBits % BITS_PER_UNIT;

  for (size_t i = 0; i < fullUnits; i++) {
    if (buf[i] != (unit_t)0)
      return false;
  }
  if (remainder > 0) {
    unit_t mask = ((unit_t)1 << remainder) - (unit_t)1;
    if ((buf[fullUnits] & mask) != 0)
      return false;
  }
  return true;
}

inline int bitCollection_size(
    unit_t *buf,
    size_t validBits) { // only works on 64 bits....need to replace
                        // __builtin_popcountll
  size_t fullUnits = validBits / BITS_PER_UNIT;
  size_t remainder = validBits % BITS_PER_UNIT;
  int count = 0;

  for (size_t i = 0; i < fullUnits; i++) {
    count += __builtin_popcountll(buf[i]);
  }

  if (remainder > 0) {
    unit_t mask = ((unit_t)1 << remainder) - (unit_t)1;
    count += __builtin_popcountll(buf[fullUnits] & mask);
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
