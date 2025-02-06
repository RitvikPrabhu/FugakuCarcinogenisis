#pragma once

#include <climits>
#include <cstring>
#include <vector>
#include <set>
#include <cstdint>
#include <cassert>


#define USE_BITWISE_SETS

#ifndef USE_BITWISE_SETS
// SET et al.
typedef std::set<int> SET;
typedef std::vector<std::set<int>> SETS;
#define SET_NEW() {}
#define SET_INSERT(set, idx) set.insert((idx))
#define SET_TEST(set, idx) bool(set.find(idx) != set.end())
#else // USE_BITWISE_SETS
// BSET et al.
typedef std::int64_t unit_t;
typedef unit_t *SET;
typedef unit_t *SETS;
#define BITS_PER_UNIT (sizeof(unit_t) * CHAR_BIT)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define UNIT_IDX(idx) (idx/BITS_PER_UNIT)
#define UNIT_MASK(idx) (((unit_t)1) << (idx % BITS_PER_UNIT))
//
#define SET_CLEAR(set, size_in_bits) {                                 \
    size_t size = CEIL_DIV(size_in_bits, BITS_PER_UNIT);                \
    for(size_t i = 0; i < size; i++) set[i] = 0;                        \
    }
#define SET_NEW(set, size_in_bits) {                                   \
    set = new unit_t[CEIL_DIV(size_in_bits, BITS_PER_UNIT)];            \
    BSET_CLEAR(set, size_in_bits);                                      \
  }
#define BSET_INSERT(set, idx) set[UNIT_IDX(idx)] |= UNIT_MASK(idx)
#define BSET_TEST(set, idx) bool(set[UNIT_IDX(idx)] & UNIT_MASK(idx))

#endif
