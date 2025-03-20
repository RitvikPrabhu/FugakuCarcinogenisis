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

void initialize_mpi(int argc, char** argv, int& size, int& rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

int64_t get_quadruplet_count(const char* filename, int rank) {
    int64_t quadruplet_count = 0;
    if (rank == 0) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Could not open binary file " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.read(reinterpret_cast<char*>(&quadruplet_count), sizeof(int64_t));
        if (infile.fail()) {
            std::cerr << "Error reading quadruplet count from binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.close();
    }
    MPI_Bcast(&quadruplet_count, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    return quadruplet_count;
}

void calculate_indices(int rank, int size, int64_t total_count, int64_t& start_idx, int64_t& end_idx) {
    int64_t per_process = total_count / size;
    int64_t remainder = total_count % size;
    start_idx = rank * per_process + std::min<int64_t>(rank, remainder);
    end_idx = start_idx + per_process + (rank < remainder ? 1 : 0);
}

std::vector<std::array<int32_t, 4>> read_quadruplets_segment(const char* filename, int64_t start_quadruplet, int64_t end_quadruplet) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open binary file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Skip the first 8 bytes (int64_t quadruplet count)
    infile.seekg(8 + start_quadruplet * 16, std::ios::beg);
    if (infile.fail()) {
        std::cerr << "Error seeking in binary file" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int64_t num_quadruplets_to_read = end_quadruplet - start_quadruplet;
    std::vector<std::array<int32_t, 4>> local_quadruplets(num_quadruplets_to_read);

    for (int64_t i = 0; i < num_quadruplets_to_read; ++i) {
        int32_t quadruplet[4];
        infile.read(reinterpret_cast<char*>(quadruplet), sizeof(int32_t) * 4);
        if (infile.fail()) {
            std::cerr << "Error reading quadruplet from binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_quadruplets[i][0] = quadruplet[0];
        local_quadruplets[i][1] = quadruplet[1];
        local_quadruplets[i][2] = quadruplet[2];
        local_quadruplets[i][3] = quadruplet[3];
    }

    infile.close();

    return local_quadruplets;
}

std::pair<double, std::array<int32_t, 4>> maxF(
    std::vector<std::set<int>>& tumorData,
    std::vector<std::set<int>>& normalData,
    int Nt,
    int Nn,
    double alpha,
    const std::vector<std::array<int32_t, 4>>& local_quadruplets
) {
    double maxF = -1.0;
    std::array<int32_t, 4> bestComb = {-1, -1, -1, -1};
    for (const auto& quadruplet : local_quadruplets) {
        int i = quadruplet[0];
        int j = quadruplet[1];
        int k = quadruplet[2];
        int l = quadruplet[3];

        const std::set<int>& gene1Tumor = tumorData[i];
        const std::set<int>& gene2Tumor = tumorData[j];
        const std::set<int>& gene3Tumor = tumorData[k];
        const std::set<int>& gene4Tumor = tumorData[l];

        std::set<int> intersectTumor1;
        std::set<int> intersectTumor2;
        std::set<int> intersectTumor3;
        std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(),
                              std::inserter(intersectTumor1, intersectTumor1.begin()));
        std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(),
                              std::inserter(intersectTumor2, intersectTumor2.begin()));
        std::set_intersection(gene4Tumor.begin(), gene4Tumor.end(), intersectTumor2.begin(), intersectTumor2.end(),
                              std::inserter(intersectTumor3, intersectTumor3.begin()));

        const std::set<int>& gene1Normal = normalData[i];
        const std::set<int>& gene2Normal = normalData[j];
        const std::set<int>& gene3Normal = normalData[k];
        const std::set<int>& gene4Normal = normalData[l];

        std::set<int> intersectNormal1;
        std::set<int> intersectNormal2;
        std::set<int> intersectNormal3;
        std::set_intersection(gene1Normal.begin(), gene1Normal.end(), gene2Normal.begin(), gene2Normal.end(),
                              std::inserter(intersectNormal1, intersectNormal1.begin()));
        std::set_intersection(gene3Normal.begin(), gene3Normal.end(), intersectNormal1.begin(), intersectNormal1.end(),
                              std::inserter(intersectNormal2, intersectNormal2.begin()));
        std::set_intersection(gene4Normal.begin(), gene4Normal.end(), intersectNormal2.begin(), intersectNormal2.end(),
                              std::inserter(intersectNormal3, intersectNormal3.begin()));

        int TP = intersectTumor3.size();
        int FP = intersectNormal3.size();

        int TN = Nn - FP;
        int FN = Nt - TP;

        double F = (alpha * TP + TN) / static_cast<double>(Nt + Nn);
        if (F >= maxF) {
            maxF = F;
            bestComb = {i, j, k};
        }
    }

    return std::make_pair(maxF, bestComb);
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

    auto get_quadruplet_count_start = std::chrono::high_resolution_clock::now();
    int64_t quadruplet_count = get_quadruplet_count(argv[1], rank);
    auto get_quadruplet_count_end = std::chrono::high_resolution_clock::now();

    auto calc_indices_start = std::chrono::high_resolution_clock::now();
    int64_t start_quadruplet, end_quadruplet;
    calculate_indices(rank, size, quadruplet_count, start_quadruplet, end_quadruplet);
    auto calc_indices_end = std::chrono::high_resolution_clock::now();

    auto read_quadruplets_segment_start = std::chrono::high_resolution_clock::now();
    std::vector<std::array<int32_t, 4>> local_quadruplets = read_quadruplets_segment(argv[1], start_quadruplet, end_quadruplet);
    auto read_quadruplets_segment_end = std::chrono::high_resolution_clock::now();

    double alpha = 0.1;

    std::set<int> droppedSamples;
    int numComb = 0;

    auto main_loop_start = std::chrono::high_resolution_clock::now();
    while (tumorSamples != droppedSamples) {
        std::pair<double, std::array<int32_t, 4>> maxFOut = maxF(tumorData, normalData, numTumor, numNormal, alpha, local_quadruplets);
        double localMaxF = maxFOut.first;
        std::array<int32_t, 4> localBestComb = maxFOut.second;

        struct {
            double value;
            int rank;
        } localResult, globalResult;

        localResult.value = localMaxF;
        localResult.rank = rank;

        MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        std::array<int32_t, 4> globalBestComb;
        if (rank == globalResult.rank) {
            globalBestComb = localBestComb;
        }

        MPI_Bcast(globalBestComb.data(), 4, MPI_INT, globalResult.rank, MPI_COMM_WORLD);
        std::set<int> finalIntersect1;
        std::set<int> finalIntersect2;
        std::set<int> sampleToCover;
        std::set_intersection(tumorData[globalBestComb[0]].begin(), tumorData[globalBestComb[0]].end(),
                              tumorData[globalBestComb[1]].begin(), tumorData[globalBestComb[1]].end(),
                              std::inserter(finalIntersect1, finalIntersect1.begin()));
        std::set_intersection(finalIntersect1.begin(), finalIntersect1.end(),
                              tumorData[globalBestComb[2]].begin(), tumorData[globalBestComb[2]].end(),
                              std::inserter(finalIntersect2, finalIntersect2.begin()));
        std::set_intersection(finalIntersect2.begin(), finalIntersect2.end(),
                              tumorData[globalBestComb[3]].begin(), tumorData[globalBestComb[3]].end(),
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
    std::chrono::duration<double> get_quadruplet_count_time = get_quadruplet_count_end - get_quadruplet_count_start;
    std::chrono::duration<double> calc_indices_time = calc_indices_end - calc_indices_start;
    std::chrono::duration<double> read_quadruplets_segment_time = read_quadruplets_segment_end - read_quadruplets_segment_start;
    std::chrono::duration<double> main_loop_time = main_loop_end - main_loop_start;

    // Only rank 0 writes to output.txt
    if (rank == 0) {
        //std::ofstream outfile(argv[3], std::ios::app);
		std::ostream& outfile = std::cout;
        //if (!outfile) {
        //    std::cerr << "Error: Could not open output file." << std::endl;
        //    MPI_Abort(MPI_COMM_WORLD, 1);
        //}

        outfile << "----------------------Timing Information for 3-hit set cover----------------------" << std::endl;
        outfile << "Total time: " << total_time.count() << " seconds" << std::endl;
        outfile << "Time for reading data: " << read_data_time.count() << " seconds" << std::endl;
        outfile << "Time for getting quadruplet count: " << get_quadruplet_count_time.count() << " seconds" << std::endl;
        outfile << "Time for calculating indices: " << calc_indices_time.count() << " seconds" << std::endl;
        outfile << "Time for reading quadruplets segment: " << read_quadruplets_segment_time.count() << " seconds" << std::endl;
        outfile << "Time for main loop: " << main_loop_time.count() << " seconds" << std::endl;
        outfile << std::endl;
       // outfile.close();
    }
    return 0;
}

