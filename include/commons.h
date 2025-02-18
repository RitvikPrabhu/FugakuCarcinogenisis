#ifndef COMMONS_H
#define COMMONS_H

#ifdef USE_CPP_SET
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

typedef std::set<int> SET;
typedef std::vector<std::set<int>> SET_COLLECTION;

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  SET_COLLECTION tumorData;
  SET_COLLECTION normalData;
};

#define BITS_PER_UNIT 1

#define ALL_BITS_SET 1

#define UNITS_FOR_BITS(N) 1

#define CALCULATE_UNITS(numSample) numSample

#define INIT_DATA(TABLE)                                                       \
  do {                                                                         \
    (TABLE).tumorData = SET_COLLECTION((TABLE).numRows);                       \
    (TABLE).normalData = SET_COLLECTION((TABLE).numRows);                      \
  } while (0)

#define SET_TUMOR(TABLE, ROW_INDEX, C)                                         \
  (TABLE).tumorData[(ROW_INDEX)].insert((C));

#define SET_NORMAL(TABLE, ROW_INDEX, C)                                        \
  (TABLE).normalData[(ROW_INDEX)].insert((C));

#else
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

typedef uint64_t unit_t;
#define MPI_UNIT_T MPI_UINT64_T

typedef unit_t SET;
typedef unit_t *SET_COLLECTION;

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  SET_COLLECTION tumorData;
  SET_COLLECTION normalData;
};

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)

#define ALL_BITS_SET (~(unit_t)0)

#define UNITS_FOR_BITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

#define CALCULATE_UNITS(numSample)                                             \
  (((numSample) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

#define INIT_DATA(TABLE)                                                       \
  do {                                                                         \
    size_t tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                  \
    size_t normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);                \
    size_t totalTumorUnits = tumorRowUnits * (TABLE).numRows;                  \
    size_t totalNormalUnits = normalRowUnits * (TABLE).numRows;                \
    (TABLE).tumorData = new SET[totalTumorUnits];                              \
    (TABLE).normalData = new SET[totalNormalUnits];                            \
    std::memset((TABLE).tumorData, 0, totalTumorUnits * sizeof(SET));          \
    std::memset((TABLE).normalData, 0, totalNormalUnits * sizeof(SET));        \
  } while (0)

inline void set_bit(unit_t *array, size_t row, size_t col,
                    size_t row_size_in_bits) {
  size_t idx = row * row_size_in_bits + col;
  size_t unit_idx = idx / BITS_PER_UNIT;
  size_t bit_in_unit = idx % BITS_PER_UNIT;
  array[unit_idx] |= ((unit_t)1 << bit_in_unit);
}

#define SET_TUMOR(TABLE, ROW_INDEX, C)                                         \
  do {                                                                         \
    size_t __tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                \
    size_t __tumorRowSizeBits = (__tumorRowUnits) * BITS_PER_UNIT;             \
    set_bit((TABLE).tumorData, (ROW_INDEX), (C), __tumorRowSizeBits);          \
  } while (0)

#define SET_NORMAL(TABLE, ROW_INDEX, C)                                        \
  do {                                                                         \
    size_t __normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);              \
    size_t __normalRowSizeBits = (__normalRowUnits) * BITS_PER_UNIT;           \
    size_t __colInNormal = (C) - (TABLE).numTumor;                             \
    set_bit((TABLE).normalData, (ROW_INDEX), __colInNormal,                    \
            __normalRowSizeBits);                                              \
  } while (0)

#endif
#endif
