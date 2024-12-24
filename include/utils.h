#ifndef UTILS_H
#define UTILS_H
#include <set>
#include <utility>
#include <vector>

long long int nCr(int n, int r);
void write_timings_to_file(const double all_times[][6], int size,
                           long long int totalCount, const char *filename);
std::string *read_data(const char *filename, int &numGenes, int &numSamples,
                       int &numTumor, int &numNormal,
                       std::set<int> &tumorSamples,
                       std::vector<std::set<int>> &sparseTumorData,
                       std::vector<std::set<int>> &sparseNormalData, int rank);
#endif
