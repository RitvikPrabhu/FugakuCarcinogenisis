#ifndef COMMONS_H
#define COMMONS_H

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#ifdef USE_CPP_SET

typedef std::set<int> SET;
typedef std::vector<SET> SET_COLLECTION;

#define SET_NEW(set, size_in_bits)                                             \
  {                                                                            \
  }

#define SET_COLLECTION_NEW(collection, rowCount, colCount)                     \
  do {                                                                         \
    (collection).resize(rowCount);                                             \
  } while (0)

#define SET_INSERT(set, idx) ((set).insert((idx)))
#define SET_COLLECTION_INSERT(collection, row, col, rowWidth)                  \
  do {                                                                         \
    SET_INSERT((collection)[(row)], (col));                                    \
  } while (0)

#define SET_TEST(set, idx) ((set).find(idx) != (set).end())

#else

typedef int64_t unit_t;
typedef unit_t *SET_COLLECTION;

#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define SET_NEW(set, size_in_bits)                                             \
  do {                                                                         \
    (set) = new unit_t[CEIL_DIV((size_in_bits), BITS_PER_UNIT)]();             \
  } while (0)

#define SET_COLLECTION_NEW(collection, rowCount, colCount)                     \
  do {                                                                         \
    size_t totalBits = (rowCount) * (colCount);                                \
    SET_NEW((collection), totalBits);                                          \
  } while (0)

#define SET_INSERT(set, idx)                                                   \
  ((set)[(idx) / BITS_PER_UNIT] |= ((unit_t)1 << ((idx) % BITS_PER_UNIT)))

#define SET_COLLECTION_INSERT(collection, row, col, rowWidth)                  \
  do {                                                                         \
    size_t offset = (row) * (rowWidth) + (col);                                \
    SET_INSERT((collection), offset);                                          \
  } while (0)

#define SET_TEST(set, idx)                                                     \
  (((set)[(idx) / BITS_PER_UNIT] & ((unit_t)1 << ((idx) % BITS_PER_UNIT))) != 0)

#endif

struct sets_t {
  size_t numRows;
  size_t numTumor;
  size_t numNormal;
  size_t numCols;
  SET_COLLECTION tumorData;
  SET_COLLECTION normalData;
};

#endif
