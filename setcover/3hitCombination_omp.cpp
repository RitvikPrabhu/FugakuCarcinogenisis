#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <climits>
#include <omp.h>

void initialize_mpi(int argc, char** argv, int& size, int& rank) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Requested MPI_THREAD_FUNNELED not supported." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}


int64_t get_triplet_count(const char* filename, int rank) {
    int64_t triplet_count = 0;
    if (rank == 0) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Could not open binary file " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.read(reinterpret_cast<char*>(&triplet_count), sizeof(int64_t));
        if (infile.fail()) {
            std::cerr << "Error reading triplet count from binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.close();
    }
    MPI_Bcast(&triplet_count, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    return triplet_count;
}

void calculate_indices(int rank, int size, int64_t total_count, int64_t& start_idx, int64_t& end_idx) {
    int64_t per_process = total_count / size;
    int64_t remainder = total_count % size;
    start_idx = rank * per_process + std::min<int64_t>(rank, remainder);
    end_idx = start_idx + per_process + (rank < remainder ? 1 : 0);
}

std::vector<std::array<int32_t, 3>> read_triplets_segment(const char* filename, int64_t start_triplet, int64_t end_triplet) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open binary file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Skip the first 8 bytes (int64_t triplet count)
    infile.seekg(8 + start_triplet * 12, std::ios::beg);
    if (infile.fail()) {
        std::cerr << "Error seeking in binary file" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int64_t num_triplets_to_read = end_triplet - start_triplet;
    std::vector<std::array<int32_t, 3>> local_triplets(num_triplets_to_read);

    for (int64_t i = 0; i < num_triplets_to_read; ++i) {
        int32_t triplet[3];
        infile.read(reinterpret_cast<char*>(triplet), sizeof(int32_t) * 3);
        if (infile.fail()) {
            std::cerr << "Error reading triplet from binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_triplets[i][0] = triplet[0];
        local_triplets[i][1] = triplet[1];
        local_triplets[i][2] = triplet[2];
    }

    infile.close();

    return local_triplets;
}

std::pair<double, std::array<int32_t, 3>> maxF(
    std::vector<std::set<int>>& tumorData,
    std::vector<std::set<int>>& normalData,
    int Nt,
    int Nn,
    double alpha,
    const std::vector<std::array<int32_t, 3>>& local_triplets
) {
    double maxF_global = -1.0;
    std::array<int32_t, 3> bestComb_global = {-1, -1, -1};

#pragma omp parallel
    {
        double maxF_local = -1.0;
        std::array<int32_t, 3> bestComb_local = {-1, -1, -1};

#pragma omp for schedule(static) nowait
        for (size_t idx = 0; idx < local_triplets.size(); ++idx) {
            const auto& triplet = local_triplets[idx];
            int i = triplet[0];
            int j = triplet[1];
            int k = triplet[2];

            // Tumor intersections
            const std::set<int>& gene1Tumor = tumorData[i];
            const std::set<int>& gene2Tumor = tumorData[j];
            const std::set<int>& gene3Tumor = tumorData[k];

            std::set<int> intersectTumor1;
            std::set<int> intersectTumor2;
            std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(),
                                  std::inserter(intersectTumor1, intersectTumor1.begin()));
            std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(),
                                  std::inserter(intersectTumor2, intersectTumor2.begin()));

            // Normal intersections
            const std::set<int>& gene1Normal = normalData[i];
            const std::set<int>& gene2Normal = normalData[j];
            const std::set<int>& gene3Normal = normalData[k];

            std::set<int> intersectNormal1;
            std::set<int> intersectNormal2;
            std::set_intersection(gene1Normal.begin(), gene1Normal.end(), gene2Normal.begin(), gene2Normal.end(),
                                  std::inserter(intersectNormal1, intersectNormal1.begin()));
            std::set_intersection(gene3Normal.begin(), gene3Normal.end(), intersectNormal1.begin(), intersectNormal1.end(),
                                  std::inserter(intersectNormal2, intersectNormal2.begin()));

            // Calculating metrics
            int TP = intersectTumor2.size();
            int FP = intersectNormal2.size();
            int TN = Nn - FP;
            int FN = Nt - TP;

            double F = (alpha * TP + TN) / static_cast<double>(Nt + Nn);
            if (F > maxF_local) {
                maxF_local = F;
                bestComb_local = {i, j, k};
            }
        }

#pragma omp critical
        {
            if (maxF_local > maxF_global) {
                maxF_global = maxF_local;
                bestComb_global = bestComb_local;
            }
        }
    }

    return std::make_pair(maxF_global, bestComb_global);
}

std::string* read_data(
    const char* filename,
    int& numGenes,
    int& numSamples,
    int& numTumor,
    int& numNormal,
    std::set<int>& tumorSamples,
    std::vector<std::set<int>>& sparseTumorData,
    std::vector<std::set<int>>& sparseNormalData
) {
    FILE* dataFile;
    dataFile = fopen(filename, "r");

    if (dataFile == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        exit(1);
    }

    char geneId[1024], sampleId[1024];
    int value;
    if (fscanf(dataFile, "%d %d %d %d %d\n", &numGenes, &numSamples, &value, &numTumor, &numNormal) != 5) {
        printf("Error reading the first line numbers\n");
        fclose(dataFile);
        MPI_Finalize();
        exit(1);
    }

    sparseTumorData.resize(numGenes);
    sparseNormalData.resize(numGenes);

    int fileRows = numGenes * numSamples;
    std::string* geneIdArray = new std::string[numGenes];
    for (int i = 0; i < fileRows; i++) {
        int gene, sample;
        if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
            printf("Error reading the line numbers\n");
            fclose(dataFile);
            MPI_Finalize();
            exit(1);
        }

        geneIdArray[gene] = geneId;

        if (value > 0) {
            if (sample < numTumor) {
                sparseTumorData[gene].insert(sample);
                tumorSamples.insert(sample);
            } else {
                sparseNormalData[gene].insert(sample);
            }
        }
    }

    fclose(dataFile);
    return geneIdArray;
}

int main(int argc, char** argv) {
    auto total_start = std::chrono::high_resolution_clock::now();
    int size, rank;
    initialize_mpi(argc, argv, size, rank);
    auto mpi_init_end = std::chrono::high_resolution_clock::now();

    if (argc != 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <intermediate binary data file> <raw input file> <metrics output file> <results file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    auto read_data_start = std::chrono::high_resolution_clock::now();
    int numGenes, numSamples, numTumor, numNormal;
    std::set<int> tumorSamples;
    std::vector<std::set<int>> tumorData;
    std::vector<std::set<int>> normalData;
    std::string* geneIdArray = read_data(argv[2], numGenes, numSamples, numTumor, numNormal, tumorSamples, tumorData, normalData);
    auto read_data_end = std::chrono::high_resolution_clock::now();

    auto get_triplet_count_start = std::chrono::high_resolution_clock::now();
    int64_t triplet_count = get_triplet_count(argv[1], rank);
    auto get_triplet_count_end = std::chrono::high_resolution_clock::now();

    auto calc_indices_start = std::chrono::high_resolution_clock::now();
    int64_t start_triplet, end_triplet;
    calculate_indices(rank, size, triplet_count, start_triplet, end_triplet);
    auto calc_indices_end = std::chrono::high_resolution_clock::now();

    auto read_triplets_segment_start = std::chrono::high_resolution_clock::now();
    std::vector<std::array<int32_t, 3>> local_triplets = read_triplets_segment(argv[1], start_triplet, end_triplet);
    auto read_triplets_segment_end = std::chrono::high_resolution_clock::now();

    double alpha = 0.1;

    std::set<int> droppedSamples;
    int numComb = 0;

    auto main_loop_start = std::chrono::high_resolution_clock::now();
    while (tumorSamples != droppedSamples) {
        std::pair<double, std::array<int32_t, 3>> maxFOut = maxF(tumorData, normalData, numTumor, numNormal, alpha, local_triplets);
        double localMaxF = maxFOut.first;
        std::array<int32_t, 3> localBestComb = maxFOut.second;

        struct {
            double value;
            int rank;
        } localResult, globalResult;

        localResult.value = localMaxF;
        localResult.rank = rank;

        MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        std::array<int32_t, 3> globalBestComb;
        if (rank == globalResult.rank) {
            globalBestComb = localBestComb;
        }

        MPI_Bcast(globalBestComb.data(), 3, MPI_INT, globalResult.rank, MPI_COMM_WORLD);
        std::set<int> finalIntersect1;
        std::set<int> sampleToCover;
        std::set_intersection(tumorData[globalBestComb[0]].begin(), tumorData[globalBestComb[0]].end(),
                              tumorData[globalBestComb[1]].begin(), tumorData[globalBestComb[1]].end(),
                              std::inserter(finalIntersect1, finalIntersect1.begin()));
        std::set_intersection(finalIntersect1.begin(), finalIntersect1.end(),
                              tumorData[globalBestComb[2]].begin(), tumorData[globalBestComb[2]].end(),
                              std::inserter(sampleToCover, sampleToCover.begin()));

        droppedSamples.insert(sampleToCover.begin(), sampleToCover.end());

        for (auto& tumorSet : tumorData) {
            for (const int sample : sampleToCover) {
                tumorSet.erase(sample);
            }
        }
        numTumor -= sampleToCover.size();

        if (rank == 0) {
            std::ofstream outfile(argv[4], std::ios::app);
            if (!outfile) {
                std::cerr << "Error: Could not open output file." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            outfile << numComb << "- (";
            for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
                outfile << geneIdArray[globalBestComb[idx]];
                if (idx != globalBestComb.size() - 1) {
                    outfile << ", ";
                }
            }
            outfile << ")  F-max = " << globalResult.value << std::endl;
            outfile.close();
            numComb++;
        }
    }
    auto main_loop_end = std::chrono::high_resolution_clock::now();
    MPI_Finalize();
    auto mpi_finalize_end = std::chrono::high_resolution_clock::now();
    delete[] geneIdArray;

    auto total_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> total_time = total_end - total_start;
    std::chrono::duration<double> read_data_time = read_data_end - read_data_start;
    std::chrono::duration<double> get_triplet_count_time = get_triplet_count_end - get_triplet_count_start;
    std::chrono::duration<double> calc_indices_time = calc_indices_end - calc_indices_start;
    std::chrono::duration<double> read_triplets_segment_time = read_triplets_segment_end - read_triplets_segment_start;
    std::chrono::duration<double> main_loop_time = main_loop_end - main_loop_start;

    // Only rank 0 writes to output.txt
    if (rank == 0) {
        std::ofstream outfile(argv[3], std::ios::app);
        if (!outfile) {
            std::cerr << "Error: Could not open output file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outfile << "----------------------Timing Information for 3-hit set cover----------------------" << std::endl;
        outfile << "Total time: " << total_time.count() << " seconds" << std::endl;
        outfile << "Time for reading data: " << read_data_time.count() << " seconds" << std::endl;
        outfile << "Time for getting triplet count: " << get_triplet_count_time.count() << " seconds" << std::endl;
        outfile << "Time for calculating indices: " << calc_indices_time.count() << " seconds" << std::endl;
        outfile << "Time for reading triplets segment: " << read_triplets_segment_time.count() << " seconds" << std::endl;
        outfile << "Time for main loop: " << main_loop_time.count() << " seconds" << std::endl;
        outfile << std::endl;
        outfile.close();
    }
    return 0;
}

