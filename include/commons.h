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
    (dest) = (A);                                                              \
    for (auto it = (dest).begin(); it != (dest).end();) {                      \
      if ((B).find(*it) == (B).end()) {                                        \
        it = (dest).erase(it);                                                 \
      } else {                                                                 \
        ++it;                                                                  \
      }                                                                        \
    }                                                                          \
  } while (0)

#define SET_IS_EMPTY(set, size_in_bits) ((set).empty())

#define SET_COUNT(set, size_in_bits) ((int)((set).size()))

#define CHECK_ALL_BITS_SET(set, size_in_bits)                                  \
  ((int)((set).size()) == (size_in_bits))

#define SET_COPY(dest, src, size_in_bits)                                      \
  do {                                                                         \
    (dest) = (src);                                                            \
  } while (0)

#define SET_UNION(dest, A, B, size_in_bits)                                    \
  do {                                                                         \
    for (const auto &elem : (B)) {                                             \
      (dest).insert(elem);                                                     \
    }                                                                          \
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

#elif defined(USE_SVE_BIT)
#include <arm_sve.h>
typedef int64_t unit_t;

typedef unit_t *SET;
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

#define GET_ROW(dataCollection, rowIndex, rowUnits)                            \
  ((dataCollection) + ((rowIndex) * (rowUnits)))

#define SET_INTERSECT(dest, A, B, size_in_bits)                                \
  do {                                                                         \
    size_t __units = CEIL_DIV((size_in_bits), BITS_PER_UNIT);                  \
    size_t __i = 0;                                                            \
    while (__i < __units) {                                                    \
      svbool_t __pg = svwhilelt_b64(__i, __units);                             \
      svuint64_t __va =                                                        \
          svld1(__pg, reinterpret_cast<const uint64_t *>(&(A)[__i]));          \
      svuint64_t __vb =                                                        \
          svld1(__pg, reinterpret_cast<const uint64_t *>(&(B)[__i]));          \
      svuint64_t __vr = svand_z(__pg, __va, __vb);                             \
      svst1(__pg, reinterpret_cast<uint64_t *>(&(dest)[__i]), __vr);           \
      __i += svcntd();                                                         \
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
    uint64_t __totalCount = 0;                                                 \
    size_t __i = 0;                                                            \
    while (__i < __units) {                                                    \
      svbool_t __pg = svwhilelt_b64(__i, __units);                             \
      svuint64_t __v64 =                                                       \
          svld1(__pg, reinterpret_cast<const uint64_t *>(&(set)[__i]));        \
      svuint8_t __asBytes = svreinterpret_u8(__v64);                           \
      svuint8_t __popc = svcnt_u8_x(__pg, __asBytes);                          \
      uint64_t __partial = svaddv_u8(__pg, __popc);                            \
      __totalCount += __partial;                                               \
      __i += svcntd();                                                         \
    }                                                                          \
    __count = static_cast<int>(__totalCount);                                  \
    return __count;                                                            \
  }())

#define CHECK_ALL_BITS_SET(set, size_in_bits)                                  \
  (SET_COUNT(set, size_in_bits) == (size_in_bits))

#define SET_COPY(dest, src, size_in_bits)                                      \
  memcpy(dest, src, CEIL_DIV(size_in_bits, BITS_PER_UNIT) * sizeof(unit_t))

#define SET_UNION(dest, A, B, size_in_bits)                                    \
  do {                                                                         \
    size_t __units = CEIL_DIV((size_in_bits), BITS_PER_UNIT);                  \
    size_t __i = 0;                                                            \
    while (__i < __units) {                                                    \
      svbool_t __pg = svwhilelt_b64(__i, __units);                             \
      svuint64_t __va =                                                        \
          svld1(__pg, reinterpret_cast<const uint64_t *>(&(A)[__i]));          \
      svuint64_t __vb =                                                        \
          svld1(__pg, reinterpret_cast<const uint64_t *>(&(B)[__i]));          \
      svuint64_t __vr = svorr_z(__pg, __va, __vb);                             \
      svst1(__pg, reinterpret_cast<uint64_t *>(&(dest)[__i]), __vr);           \
      __i += svcntd();                                                         \
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

#else

typedef int64_t unit_t;

typedef unit_t *SET;
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

#define GET_ROW(dataCollection, rowIndex, rowUnits)                            \
  ((dataCollection) + ((rowIndex) * (rowUnits)))

#define SET_INTERSECT(dest, A, B, size_in_units)                               \
  do {                                                                         \
    for (size_t __i = 0; __i < size_in_units; ++__i) {                         \
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

#define CHECK_ALL_BITS_SET(set, size_in_bits)                                  \
  (SET_COUNT(set, size_in_bits) == (size_in_bits))

#define SET_COPY(dest, src, size_in_bits)                                      \
  memcpy(dest, src, CEIL_DIV(size_in_bits, BITS_PER_UNIT) * sizeof(unit_t))

#define SET_UNION(dest, A, B, size_in_bits)                                    \
  for (size_t __i = 0, __units = CEIL_DIV(size_in_bits, BITS_PER_UNIT);        \
       __i < __units; ++__i)                                                   \
  (dest)[__i] = (A)[__i] | (B)[__i]

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
