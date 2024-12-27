#ifndef UTILS_H
#define UTILS_H
#include <bitset>
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
std::string to_binary_string(unsigned long long value, int bits);
unsigned long long *allocate_bit_array(size_t units,
                                       unsigned long long init_value);
void bitwise_and_arrays(unsigned long long *result,
                        const unsigned long long *source, size_t units);
unsigned long long *get_intersection(unsigned long long **data, int numSamples,
                                     ...);
bool is_empty(unsigned long long *bitArray, size_t units);
size_t bitCollection_size(unsigned long long *bitArray, size_t units);
bool arrays_equal(const unsigned long long *a, const unsigned long long *b,
                  size_t units);
double compute_F(int TP, int TN, double alpha, int Nt, int Nn);
void update_tumor_data(unsigned long long **&tumorData,
                       unsigned long long *sampleToCover, size_t units,
                       int numGenes);
void outputFileWriteError(std::ofstream &outfile);
std::pair<long long int, long long int>
calculate_initial_chunk(int rank, long long int num_Comb,
                        long long int chunk_size);
void update_dropped_samples(unsigned long long *&droppedSamples,
                            unsigned long long *sampleToCover, size_t units);
unsigned long long *initialize_dropped_samples(size_t units);
void updateNt(int &Nt, unsigned long long *&sampleToCover);
#endif
