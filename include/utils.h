#ifndef UTILS_H
#define UTILS_H
#include <set>
#include <string>
#include <utility>
#include <vector>

#define BITS_PER_UNIT 64
#define CALCULATE_BIT_UNITS(numSample)                                         \
  (((numSample) + BITS_PER_UNIT - 1) / BITS_PER_UNIT)

void write_timings_to_file(const double all_times[][6], int size,
                           const char *filename);
std::string *read_data(const char *filename, int &numGenes, int &numSamples,
                       int &numTumor, int &numNormal,
                       unsigned long long *&tumorSamples,
                       unsigned long long **&sparseTumorData,
                       unsigned long long **&sparseNormalData, int rank);
#endif
