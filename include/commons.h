#ifndef COMMONS_H
#define COMMONS_H

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <set>
#include <vector>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#ifdef USE_CPP_SET

typedef std::set<int> SET;
typedef std::vector<SET> SET_COLLECTION;

#define BITS_PER_UNIT 64
#define SET_NEW(set, size_in_bits)                                             \
  {                                                                            \
  }

#define SET_COLLECTION_NEW(collection, rowCount, colCount, rowUnits)           \
  do {                                                                         \
    (collection).resize(rowCount);                                             \
  } while (0)

#define SET_INSERT(set, idx) ((set).insert((idx)))
#define SET_COLLECTION_INSERT(collection, row, col, rowWidth, rowUnits)        \
  do {                                                                         \
    SET_INSERT((collection)[(row)], (col));                                    \
  } while (0)

#define GET_ROW(dataCollection, rowIndex, rowUnits) ((dataCollection)[rowIndex])

#define SET_INTERSECT(dest, A, B, size_in_bits)                                \
  do {                                                                         \
    SET temp;                                                                  \
    std::set_intersection((A).begin(), (A).end(), (B).begin(), (B).end(),      \
                          std::inserter(temp, temp.begin()));                  \
    (dest).swap(temp);                                                         \
  } while (0)

#define SET_IS_EMPTY(set, size_in_bits) ((set).empty())

#define SET_COUNT(set, size_in_bits) ((int)((set).size()))

#define CHECK_ALL_BITS_SET(set, size_in_bits, size_in_units)                   \
  ((int)((set).size()) == (size_in_bits))

#define SET_COPY(dest, src, size_in_bits)                                      \
  do {                                                                         \
    (dest) = (src);                                                            \
  } while (0)

#define SET_UNION(dest, A, B, size_in_units)                                   \
  do {                                                                         \
    SET temp;                                                                  \
    std::set_union((A).begin(), (A).end(), (B).begin(), (B).end(),             \
                   std::inserter(temp, temp.begin()));                         \
    (dest).swap(temp);                                                         \
  } while (0)

#define SET_FREE(set)                                                          \
  do {                                                                         \
  } while (0)

#define UPDATE_SET_COLLECTION(dataCollection, mask, rowCount, rowUnits)        \
  do {                                                                         \
    for (auto &tumorSet : (dataCollection)) {                                  \
      for (const int sample : (mask)) {                                        \
        tumorSet.erase(sample);                                                \
      }                                                                        \
    }                                                                          \
  } while (0)

#define FREE_DATA_TABLE(dt)                                                    \
  {                                                                            \
  }

#define SET_INTERSECT_N(dest, sets_array, num_sets, size_in_units)             \
  do {                                                                         \
    SET_COPY(dest, sets_array[0], size_in_units);                              \
    for (size_t idx = 1; idx < num_sets; ++idx) {                              \
      SET_INTERSECT(dest, dest, sets_array[idx], size_in_units);               \
    }                                                                          \
  } while (0)

#else

typedef int64_t unit_t;

typedef unit_t *SET;
typedef unit_t *SET_COLLECTION;
#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)

#define SET_NEW(set, size_in_units)                                            \
  do {                                                                         \
    (set) = new unit_t[size_in_units]();                                       \
  } while (0)

#define SET_COLLECTION_NEW(collection, rowCount, colCount, rowUnits)           \
  do {                                                                         \
    size_t totalUnits = rowUnits * (rowCount);                                 \
    SET_NEW((collection), totalUnits);                                         \
  } while (0)

#define SET_INSERT(set, idx)                                                   \
  ((set)[(idx) / BITS_PER_UNIT] |= ((unit_t)1 << ((idx) % BITS_PER_UNIT)))

#define SET_COLLECTION_INSERT(collection, row, col, rowWidth, rowUnits)        \
  do {                                                                         \
    size_t offset = (row * rowUnits * 64) + (col);                             \
    SET_INSERT((collection), offset);                                          \
  } while (0)

#define GET_ROW(dataCollection, rowIndex, rowUnits)                            \
  ((dataCollection) + ((rowIndex) * (rowUnits)))

#define SET_INTERSECT(dest, A, B, size_in_units)                               \
  do {                                                                         \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      (dest)[__i] = (A)[__i] & (B)[__i];                                       \
    }                                                                          \
  } while (0)

#define SET_IS_EMPTY(set, size_in_units)                                       \
  ([&]() {                                                                     \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      if ((set)[__i] != 0)                                                     \
        return false;                                                          \
    }                                                                          \
    return true;                                                               \
  }())

#define SET_COUNT(set, size_in_units)                                          \
  ([&]() -> int {                                                              \
    int __count = 0;                                                           \
    for (size_t __i = 0; __i < (size_in_units); ++__i) {                       \
      uint64_t __val = (set)[__i];                                             \
      while (__val) {                                                          \
        __val &= (__val - 1);                                                  \
        __count++;                                                             \
      }                                                                        \
    }                                                                          \
    return __count;                                                            \
  }())

#define CHECK_ALL_BITS_SET(set, size_in_bits, size_in_units)                   \
  (SET_COUNT(set, size_in_units) == (size_in_bits))

#define SET_COPY(dest, src, size_in_units)                                     \
  memcpy((dest), (src), (size_in_units) * sizeof(unit_t))

#define SET_UNION(dest, A, B, size_in_units)                                   \
  do {                                                                         \
    for (size_t __i = 0; __i < (size_in_units); ++__i) {                       \
      (dest)[__i] = (A)[__i] | (B)[__i];                                       \
    }                                                                          \
  } while (0)

#define SET_FREE(set) delete[] set

#define UPDATE_SET_COLLECTION(dataCollection, mask, rowCount, rowUnits)        \
  do {                                                                         \
    for (size_t row_idx = 0; row_idx < (rowCount); row_idx++) {                \
      SET currentRow = GET_ROW((dataCollection), row_idx, (rowUnits));         \
      for (size_t unit_idx = 0; unit_idx < (rowUnits); unit_idx++) {           \
        currentRow[unit_idx] &= (~(mask)[unit_idx]);                           \
      }                                                                        \
    }                                                                          \
  } while (0)

#define FREE_DATA_TABLE(dt)                                                    \
  do {                                                                         \
    SET_FREE((dt).tumorData);                                                  \
    SET_FREE((dt).normalData);                                                 \
  } while (0)

#define SET_INTERSECT_N(dest, sets_array, num_sets, size_in_units)             \
  do {                                                                         \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
      dest[__i] = sets_array[0][__i];                                          \
      for (size_t idx = 1; idx < num_sets; ++idx)                              \
        dest[__i] &= sets_array[idx][__i];                                     \
    }                                                                          \
  } while (0)

#endif

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  size_t tumorRowUnits;
  size_t normalRowUnits;
  SET_COLLECTION tumorData;
  SET_COLLECTION normalData;
};

#endif
