#ifndef COMMONS_H
#define COMMONS_H

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

#ifdef USE_CPP_SET
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
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

#define INTERSECT_TWO_ROWS(DEST, PARTIAL, ROWPTR, UNITS)                       \
  do {                                                                         \
    (DEST).clear();                                                            \
    std::set_intersection((PARTIAL).begin(), (PARTIAL).end(),                  \
                          (ROWPTR).begin(), (ROWPTR).end(),                    \
                          std::inserter((DEST), (DEST).begin()));              \
  } while (0)

#define IS_EMPTY(BUF, VALID_BITS) ((BUF).empty())

#define BIT_COLLECTION_SIZE(BUF, VALID_BITS) ((BUF).size())

#define LOAD_FIRST_TUMOR(SCRATCH, TABLE, GENE)                                 \
  do {                                                                         \
    (SCRATCH) = (TABLE).tumorData[(GENE)];                                     \
  } while (0)

#define INPLACE_INTERSECT_TUMOR(SCRATCH, TABLE, GENE)                          \
  do {                                                                         \
    /* Copy the target row from tumorData into a temporary set */              \
    SET _tmp = (TABLE).tumorData[(GENE)];                                      \
    /* For each element in SCRATCH, remove it if it is not in _tmp */          \
    for (auto it = (SCRATCH).begin(); it != (SCRATCH).end();) {                \
      if (_tmp.find(*it) == _tmp.end()) {                                      \
        it = (SCRATCH).erase(it);                                              \
      } else {                                                                 \
        ++it;                                                                  \
      }                                                                        \
    }                                                                          \
  } while (0)

#define UPDATE_DROPPED_SAMPLES(DROPPED, COVER, UNITS)                          \
  do {                                                                         \
    for (size_t _i = 0; _i < (UNITS); _i++) {                                  \
      for (const auto &val : (COVER)[_i]) {                                    \
        (DROPPED)[_i].insert(val);                                             \
      }                                                                        \
    }                                                                          \
  } while (0)

#define UPDATE_TUMOR_DATA(TUMORDATA, SAMPLE_TO_COVER, UNITS, NUMGENES)         \
  do {                                                                         \
    for (int _gene = 0; _gene < (NUMGENES); ++_gene) {                         \
      for (const auto &elem : (SAMPLE_TO_COVER)[_gene]) {                      \
        (TUMORDATA)[_gene].erase(elem);                                        \
      }                                                                        \
    }                                                                          \
  } while (0)

#else
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

typedef uint64_t unit_t;
#define MPI_UNIT_T MPI_UINT64_T

typedef unit_t *SET;
typedef SET *SET_COLLECTION;

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
/**
#define INIT_DATA(TABLE)                                                       \
  do {                                                                         \
    size_t tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                  \
    size_t normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);                \
    size_t totalTumorUnits = tumorRowUnits * (TABLE).numRows;                  \
    size_t totalNormalUnits = normalRowUnits * (TABLE).numRows;                \
    (TABLE).tumorData = new unit_t[totalTumorUnits];                           \
    (TABLE).normalData = new unit_t[totalNormalUnits];                         \
    std::memset((TABLE).tumorData, 0, totalTumorUnits * sizeof(SET));          \
    std::memset((TABLE).normalData, 0, totalNormalUnits * sizeof(SET));        \
  } while (0)**/

#define INIT_DATA(TABLE)                                                       \
  do {                                                                         \
    size_t tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                  \
    size_t normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);                \
    (TABLE).tumorData = new SET[(TABLE).numRows];                              \
    (TABLE).normalData = new SET[(TABLE).numRows];                             \
                                                                               \
    for (size_t _r = 0; _r < (TABLE).numRows; ++_r) {                          \
      (TABLE).tumorData[_r] = new unit_t[tumorRowUnits];                       \
      std::memset((TABLE).tumorData[_r], 0, tumorRowUnits * sizeof(unit_t));   \
    }                                                                          \
    for (size_t _r = 0; _r < (TABLE).numRows; ++_r) {                          \
      (TABLE).normalData[_r] = new unit_t[normalRowUnits];                     \
      std::memset((TABLE).normalData[_r], 0, normalRowUnits * sizeof(unit_t)); \
    }                                                                          \
  } while (0)

#define SET_BIT(rowPtr, bitIndex)                                              \
  do {                                                                         \
    size_t __unitIdx = (bitIndex) / BITS_PER_UNIT;                             \
    size_t __bitInUnit = (bitIndex) % BITS_PER_UNIT;                           \
    (rowPtr)[__unitIdx] |= ((unit_t)1 << __bitInUnit);                         \
  } while (0)
/**
#define SET_BIT(array, row, col, row_size_in_bits)                             \
  do {                                                                         \
    size_t _idx = (row) * (row_size_in_bits) + (col);                          \
    size_t _unit_idx = _idx / BITS_PER_UNIT;                                   \
    size_t _bit_in_unit = _idx % BITS_PER_UNIT;                                \
    (array)[_unit_idx] |= ((unit_t)1 << _bit_in_unit);                         \
  } while (0)**/
/**
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
  } while (0)**/

#define SET_TUMOR(TABLE, ROW_INDEX, C)                                         \
  do {                                                                         \
    size_t _tumorRowUnits = CALCULATE_UNITS((TABLE).numTumor);                 \
    size_t _tumorRowSizeBits = _tumorRowUnits * BITS_PER_UNIT;                 \
    unit_t *_rowPtr = (TABLE).tumorData[(ROW_INDEX)];                          \
    SET_BIT(_rowPtr, (C));                                                     \
  } while (0)

#define SET_NORMAL(TABLE, ROW_INDEX, C)                                        \
  do {                                                                         \
    size_t _normalRowUnits = CALCULATE_UNITS((TABLE).numNormal);               \
    size_t _normalRowSizeBits = _normalRowUnits * BITS_PER_UNIT;               \
    size_t _colInNormal = (C) - (TABLE).numTumor;                              \
    unit_t *_rowPtr = (TABLE).normalData[(ROW_INDEX)];                         \
    SET_BIT(_rowPtr, _colInNormal);                                            \
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

#define INTERSECT_TWO_ROWS(DEST, PARTIAL, ROWPTR, UNITS)                       \
  do {                                                                         \
    for (size_t b = 0; b < (UNITS); b++) {                                     \
      (DEST)[b] = (PARTIAL)[b] & (ROWPTR)[b];                                  \
    }                                                                          \
  } while (0)

#define IS_EMPTY(BUF, VALID_BITS)                                              \
  ([&]() -> bool {                                                             \
    size_t _fullUnits = (VALID_BITS) / BITS_PER_UNIT;                          \
    size_t _remainder = (VALID_BITS) % BITS_PER_UNIT;                          \
    for (size_t _i = 0; _i < _fullUnits; _i++) {                               \
      if ((BUF)[_i] != (unit_t)0)                                              \
        return false;                                                          \
    }                                                                          \
    if (_remainder > 0) {                                                      \
      unit_t _mask = (((unit_t)1 << _remainder) - (unit_t)1);                  \
      if (((BUF)[_fullUnits] & _mask) != 0)                                    \
        return false;                                                          \
    }                                                                          \
    return true;                                                               \
  }())

// only works on 64 bits....TODO: need to replace
// __builtin_popcountll
#define BIT_COLLECTION_SIZE(BUF, VALID_BITS)                                   \
  ([&]() -> int {                                                              \
    size_t _fullUnits = (VALID_BITS) / BITS_PER_UNIT;                          \
    size_t _remainder = (VALID_BITS) % BITS_PER_UNIT;                          \
    int _count = 0;                                                            \
    for (size_t _i = 0; _i < _fullUnits; _i++) {                               \
      _count += __builtin_popcountll((BUF)[_i]);                               \
    }                                                                          \
    if (_remainder > 0) {                                                      \
      unit_t _mask = (((unit_t)1 << _remainder) - (unit_t)1);                  \
      _count += __builtin_popcountll((BUF)[_fullUnits] & _mask);               \
    }                                                                          \
    return _count;                                                             \
  }())

#define LOAD_FIRST_TUMOR(SCRATCH, TABLE, GENE)                                 \
  do {                                                                         \
    size_t __rowUnits = UNITS_FOR_BITS((TABLE).numTumor);                      \
    size_t __baseIdx = (GENE) * __rowUnits;                                    \
    std::memcpy((SCRATCH), &((TABLE).tumorData[__baseIdx]),                    \
                __rowUnits * sizeof(unit_t));                                  \
  } while (0)

#define INPLACE_INTERSECT_TUMOR(SCRATCH, TABLE, GENE)                          \
  do {                                                                         \
    size_t __rowUnits = UNITS_FOR_BITS((TABLE).numTumor);                      \
    size_t __baseIdx = (GENE) * __rowUnits;                                    \
    for (size_t _b = 0; _b < __rowUnits; _b++) {                               \
      (SCRATCH)[_b] &= (TABLE).tumorData[__baseIdx + _b];                      \
    }                                                                          \
  } while (0)

#define UPDATE_DROPPED_SAMPLES(DROPPED, COVER, UNITS)                          \
  do {                                                                         \
    for (size_t _i = 0; _i < (UNITS); _i++) {                                  \
      (DROPPED)[_i] |= (COVER)[_i];                                            \
    }                                                                          \
  } while (0)

#define UPDATE_TUMOR_DATA(TUMORDATA, SAMPLE_TO_COVER, UNITS, NUMGENES)         \
  do {                                                                         \
    for (int _gene = 0; _gene < (NUMGENES); ++_gene) {                         \
      unit_t *_geneRow = (TUMORDATA) + _gene * (UNITS);                        \
      for (size_t _i = 0; _i < (UNITS); ++_i) {                                \
        _geneRow[_i] &= ~((SAMPLE_TO_COVER)[_i]);                              \
      }                                                                        \
    }                                                                          \
  } while (0)

#endif
#endif
