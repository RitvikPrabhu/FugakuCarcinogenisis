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
typedef long long LAMBDA_TYPE;

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

#define CALCULATE_BIT_UNITS(N) (N)

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

#define INIT_BUFFERS(INTERSECTION_BUFFER, SCRATCH_BUFFER_IJ,                   \
                     SCRATCH_BUFFER_IJK, MAX_UNITS)                            \
  do {                                                                         \
    (INTERSECTION_BUFFER) = new SET_COLLECTION((MAX_UNITS));                   \
    (SCRATCH_BUFFER_IJ) = new SET_COLLECTION((MAX_UNITS));                     \
    (SCRATCH_BUFFER_IJK) = new SET_COLLECTION((MAX_UNITS));                    \
  } while (0)

#define INIT_DROPPED_SAMPLES(X, UNITS)                                         \
  do {                                                                         \
    (X) = SET_COLLECTION((UNITS));                                             \
  } while (0)

#define CHECK_ALL_BITS_SET(DROPPED_SAMPLES, TARGET)                            \
  (std::all_of((DROPPED_SAMPLES).begin(), (DROPPED_SAMPLES).end(),             \
               [&TARGET](const SET &val) { return val == TARGET; }))

#define GET_ROW(COLLECTION, i, UNITS) ((COLLECTION)[(i)])

#else
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

typedef uint64_t unit_t;
#define MPI_UNIT_T MPI_UINT64_T

typedef unit_t SET;
typedef unit_t *SET_COLLECTION;
typedef unit_t LAMBDA_TYPE;

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

#define CALCULATE_BIT_UNITS(N) (((N) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

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

#define SET_BIT(array, row, col, row_size_in_bits)                             \
  do {                                                                         \
    size_t _idx = (row) * (row_size_in_bits) + (col);                          \
    size_t _unit_idx = _idx / BITS_PER_UNIT;                                   \
    size_t _bit_in_unit = _idx % BITS_PER_UNIT;                                \
    (array)[_unit_idx] |= ((unit_t)1 << _bit_in_unit);                         \
  } while (0)

#define SET_TUMOR(TABLE, ROW_INDEX, C)                                         \
  do {                                                                         \
    size_t __tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                \
    size_t __tumorRowSizeBits = (__tumorRowUnits) * BITS_PER_UNIT;             \
    SET_BIT((TABLE).tumorData, (ROW_INDEX), (C), __tumorRowSizeBits);          \
  } while (0)

#define SET_NORMAL(TABLE, ROW_INDEX, C)                                        \
  do {                                                                         \
    size_t __normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);              \
    size_t __normalRowSizeBits = (__normalRowUnits) * BITS_PER_UNIT;           \
    size_t __colInNormal = (C) - (TABLE).numTumor;                             \
    SET_BIT((TABLE).normalData, (ROW_INDEX), __colInNormal,                    \
            __normalRowSizeBits);                                              \
  } while (0)

#define INIT_BUFFERS(INTERSECTION_BUFFER, SCRATCH_BUFFER_IJ,                   \
                     SCRATCH_BUFFER_IJK, MAX_UNITS)                            \
  do {                                                                         \
    (INTERSECTION_BUFFER) = new SET[(MAX_UNITS)];                              \
    (SCRATCH_BUFFER_IJ) = new SET[(MAX_UNITS)];                                \
    (SCRATCH_BUFFER_IJK) = new SET[(MAX_UNITS)];                               \
  } while (0)

#define INIT_DROPPED_SAMPLES(X, UNITS)                                         \
  do {                                                                         \
    (X) = new SET[(UNITS)];                                                    \
    memset((X), 0, (UNITS) * sizeof(SET));                                     \
  } while (0)

#define CHECK_ALL_BITS_SET(DROPPED_SAMPLES, UNITS)                             \
  (std::all_of((DROPPED_SAMPLES), (DROPPED_SAMPLES) + (UNITS),                 \
               [](SET val) { return val == ~static_cast<SET>(0); }))

#define GET_ROW(COLLECTION, i, UNITS) ((COLLECTION) + ((i) * (UNITS)))

#endif
#endif
