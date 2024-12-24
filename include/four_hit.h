#ifndef FOUR_HIT_H
#define FOUR_HIT_H

#include <array>
#include <set>
#include <string>
#include <vector>

void process_lambda_interval(const std::vector<std::set<int>> &tumorData,
                             const std::vector<std::set<int>> &normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes, long long int &count,
                             std::array<int, 4> &bestCombination, int Nt,
                             int Nn, double &maxF);

void worker_process(int rank, long long int num_Comb,
                    std::vector<std::set<int>> &tumorData,
                    const std::vector<std::set<int>> &normalData, int numGenes,
                    long long int &count, int Nt, int Nn, const char *hit3_file,
                    double &localBestMaxF, std::array<int, 4> &localComb);

void distribute_tasks(int rank, int size, int numGenes,
                      std::vector<std::set<int>> &tumorData,
                      std::vector<std::set<int>> &normalData,
                      long long int &count, int Nt, int Nn,
                      const char *outFilename, const char *hit3_file,
                      const std::set<int> &tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]);

#endif
