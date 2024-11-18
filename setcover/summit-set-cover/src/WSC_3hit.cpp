#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <limits>
#include <chrono>

using namespace std;
using namespace chrono;

struct Result {
    int i, j, k;
    double Fmax;
};

double calculateF(int TP, int FP, int TN, int FN, double alpha, int Nt, int Nn) {
    return (alpha * TP + TN) / (Nt + Nn);
}

vector<tuple<int, int, int>> readCombinationsFromFile(const string &filePath) {
    ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        fprintf(stderr, "Error opening file: %s\n", filePath.c_str());
        exit(1);
    }

    vector<tuple<int, int, int>> i_j_k_values;
    int i, j, k;
    while (inputFile >> i >> j >> k) {
        i_j_k_values.push_back(make_tuple(i, j, k));
    }
    inputFile.close();
    return i_j_k_values;
}

void readSamplesFromFile(const string &filePath, vector<set<int>> &tumor_sets, vector<set<int>> &normal_sets, int &Nt, int &Nn) {
    ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        fprintf(stderr, "Error opening file: %s\n", filePath.c_str());
        exit(1);
    }

    string line;
    getline(inputFile, line);
    istringstream firstLine(line);

    int num_genes, total_samples, unused, num_tumor_samples, num_normal_samples;
    firstLine >> num_genes >> total_samples >> unused >> num_tumor_samples >> num_normal_samples;

    Nt = num_tumor_samples;
    Nn = num_normal_samples;

    tumor_sets.resize(num_genes);
    normal_sets.resize(num_genes);

    while (getline(inputFile, line)) {
        istringstream ss(line);
        int gene_index, sample_index, label;
        string gene_id, sample_id;

        ss >> gene_index >> sample_index >> label >> gene_id >> sample_id;

        if (sample_index < Nt) {
            tumor_sets[gene_index].insert(sample_index);
        } else {
            normal_sets[gene_index].insert(sample_index);
        }
    }

    inputFile.close();
}

set<int> intersectSets(const set<int> &a, const set<int> &b) {
    set<int> result;
    set_intersection(a.begin(), a.end(), b.begin(), b.end(), inserter(result, result.begin()));
    return result;
}

Result findBestCombination(const vector<tuple<int, int, int>> &i_j_k_values,
                           const vector<set<int>> &tumor_sets, const vector<set<int>> &normal_sets,
                           const set<int> &covered_samples, double alpha, int Nt, int Nn) {
    Result bestCombination = {-1, -1, -1, -numeric_limits<double>::infinity()};

    for (size_t idx = 0; idx < i_j_k_values.size(); ++idx) {
        int i = get<0>(i_j_k_values[idx]);
        int j = get<1>(i_j_k_values[idx]);
        int k = get<2>(i_j_k_values[idx]);

        set<int> tumor_intersection = intersectSets(tumor_sets[i], tumor_sets[j]);
        tumor_intersection = intersectSets(tumor_intersection, tumor_sets[k]);
        int TP = tumor_intersection.size();

        set<int> normal_intersection = intersectSets(normal_sets[i], normal_sets[j]);
        normal_intersection = intersectSets(normal_intersection, normal_sets[k]);
        int FP = normal_intersection.size();

        int TN = Nn - FP;
        int FN = Nt - TP;

        double F = calculateF(TP, FP, TN, FN, alpha, Nt, Nn);

        if (F > bestCombination.Fmax) {
            bestCombination.i = i;
            bestCombination.j = j;
            bestCombination.k = k;
            bestCombination.Fmax = F;
        }
    }

    return bestCombination;
}

void updateCoveredSamples(const vector<set<int>> &tumor_sets, const Result &bestCombination,
                          set<int> &covered_samples) {
    set<int> covered_update = intersectSets(tumor_sets[bestCombination.i], tumor_sets[bestCombination.j]);
    covered_update = intersectSets(covered_update, tumor_sets[bestCombination.k]);
    covered_samples.insert(covered_update.begin(), covered_update.end());
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <combinations_file> <samples_file>\n", argv[0]);
        return 1;
    }

    string samplesFile = argv[1];
    string combinationsFile = argv[2];

    vector<set<int>> tumor_sets, normal_sets;
    int Nt = 0, Nn = 0;

    readSamplesFromFile(samplesFile, tumor_sets, normal_sets, Nt, Nn);

    vector<tuple<int, int, int>> i_j_k_values = readCombinationsFromFile(combinationsFile);

    set<int> covered_samples;
    vector<tuple<int, int, int>> combinations;

    auto start = high_resolution_clock::now();

    while (covered_samples.size() < Nt) {
        Result bestCombination = findBestCombination(i_j_k_values, tumor_sets, normal_sets, covered_samples, 0.5, Nt, Nn);

        if (bestCombination.i != -1) {
            combinations.push_back(make_tuple(bestCombination.i, bestCombination.j, bestCombination.k));
            updateCoveredSamples(tumor_sets, bestCombination, covered_samples);
        } else {
            fprintf(stderr, "No valid combinations found!\n");
            break;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    for (size_t idx = 0; idx < combinations.size(); ++idx) {
        int i = get<0>(combinations[idx]);
        int j = get<1>(combinations[idx]);
        int k = get<2>(combinations[idx]);
        printf("Combination: i=%d, j=%d, k=%d\n", i, j, k);
    }

    printf("Execution Time: %ld milliseconds\n", duration.count());

    return 0;
}

