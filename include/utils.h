#ifndef UTILS_H
#define UTILS_H
#include <set>
#include <string>
#include <utility>
#include <vector>

long long int nCr(int n, int r);
void write_timings_to_file(const double all_times[][6], int size,
                           const char *filename);
std::string *read_data(const char *filename, int &numGenes, int &numSamples,
                       int &numTumor, int &numNormal,
                       unsigned long long *&tumorSamples,
                       unsigned long long **&sparseTumorData,
                       unsigned long long **&sparseNormalData, int rank);
size_t calculate_bit_units(size_t numGenes);
#endif
