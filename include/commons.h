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

#define GET_ROW(dataCollection, rowIndex, rowUnits) (dataCollection)[rowIndex]

#define SET_INTERSECT(dest, A, B, size_in_bits)                                \
  do {                                                                         \
    (dest).clear();                                                            \
    std::set_intersection((A).begin(), (A).end(), (B).begin(), (B).end(),      \
                          std::inserter((dest), (dest).begin()));              \
  } while (0)

#define SET_IS_EMPTY(set, size_in_bits) ((set).empty())

#define SET_COUNT(set, size_in_bits) ((int)((set).size()))

#else

typedef int64_t unit_t;
typedef unit_t *SET_COLLECTION;

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)

#define SET_NEW(set, size_in_bits)                                             \
  do {                                                                         \
    (set) = new unit_t[CEIL_DIV((size_in_bits), BITS_PER_UNIT)]();             \
  } while (0)

#define SET_COLLECTION_NEW(collection, rowCount, colCount, rowUnits)           \
  do {                                                                         \
    size_t totalUnits = rowUnits * (rowCount);                                 \
    size_t totalBits = totalUnits * BITS_PER_UNIT;                             \
    SET_NEW((collection), totalBits);                                          \
  } while (0)

#define SET_INSERT(set, idx)                                                   \
  ((set)[(idx) / BITS_PER_UNIT] |= ((unit_t)1 << ((idx) % BITS_PER_UNIT)))

#define SET_COLLECTION_INSERT(collection, row, col, rowWidth, rowUnits)        \
  do {                                                                         \
    size_t offset = (row * rowUnits * 64) + (col);                             \
    SET_INSERT((collection), offset);                                          \
  } while (0)

#define SET_TEST(set, idx)                                                     \
  (((set)[(idx) / BITS_PER_UNIT] & ((unit_t)1 << ((idx) % BITS_PER_UNIT))) != 0)

#define GET_ROW(dataCollection, rowIndex, rowUnits)                            \
  ((dataCollection) + ((rowIndex) * (rowUnits)))

#define SET_INTERSECT(dest, A, B, size_in_bits)                                \
  do {                                                                         \
    size_t __units = CEIL_DIV((size_in_bits), BITS_PER_UNIT);                  \
    for (size_t __i = 0; __i < __units; ++__i) {                               \
      (dest)[__i] = (A)[__i] & (B)[__i];                                       \
    }                                                                          \
  } while (0)

#define SET_IS_EMPTY(set, size_in_bits)                                        \
  ([&]() {                                                                     \
    size_t __units = CEIL_DIV((size_in_bits), BITS_PER_UNIT);                  \
    for (size_t __i = 0; __i < __units; ++__i) {                               \
      if ((set)[__i] != 0)                                                     \
        return false;                                                          \
    }                                                                          \
    return true;                                                               \
  }())

#define SET_COUNT(set, size_in_bits)                                           \
  ([&]() -> int {                                                              \
    size_t __units = CEIL_DIV((size_in_bits), BITS_PER_UNIT);                  \
    int __count = 0;                                                           \
    for (size_t __i = 0; __i < __units; ++__i) {                               \
      uint64_t __val = (set)[__i];                                             \
      while (__val) {                                                          \
        __val &= (__val - 1);                                                  \
        __count++;                                                             \
      }                                                                        \
    }                                                                          \
    return __count;                                                            \
  }())

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
