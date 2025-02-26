#ifndef UTILS_H
#define UTILS_H
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "carcUtils.h"

#define CALCULATE_BIT_UNITS(numSample)                                         \
  (((numSample) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

sets_t read_data(const char *filename, int rank);
#endif
