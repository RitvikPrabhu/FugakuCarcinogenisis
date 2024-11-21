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
#define MAX_BUF_SIZE 1024


void initialize_mpi(int argc, char** argv, int& size, int& rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

int get_line_count(const char* filename, int rank) {
    int line_count = 0;
    if (rank == 0) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::string line;
        while (std::getline(infile, line)) {
            line_count++;
        }
        infile.close();
    }
    MPI_Bcast(&line_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return line_count;
}

void calculate_line_indices(int rank, int size, int line_count, int& start_line, int& end_line) {
    int lines_per_process = line_count / size;
    int remainder = line_count % size;
    start_line = rank * lines_per_process + std::min(rank, remainder);
    end_line = start_line + lines_per_process + (rank < remainder ? 1 : 0);
}

std::vector<std::string> read_lines_segment(const char* filename, int start_line, int end_line) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    std::vector<std::string> local_lines;
    int current_line = 0;
    while (std::getline(infile, line)) {
        if (current_line >= start_line && current_line < end_line) {
            local_lines.push_back(line);
        }
        current_line++;
        if (current_line >= end_line) {
            break;
        }
    }
    infile.close();

    return local_lines;
}

std::pair<std::vector<std::set<int>>, std::vector<std::set<int>>> read_data(const char* filename, int& numGenes, int& numSamples, int& numTumor, int& numNormal, std::set<int>& tumorSamples) {
    FILE* dataFile;
    dataFile = fopen(filename, "r");

    if (dataFile == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        exit(1);
    }

    char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];
    int value;
    if (fscanf(dataFile, "%d %d %d %d %d\n", &numGenes, &numSamples, &value, &numTumor, &numNormal) != 5) {
        printf("Error reading the first line numbers\n");
        fclose(dataFile);
        MPI_Finalize();
        exit(1);
    }

    std::vector<std::set<int>> sparseTumorData(numGenes);
    std::vector<std::set<int>> sparseNormalData(numGenes);

    int fileRows = numGenes * numSamples;

    for (int i = 0; i < fileRows; i++){
        int gene, sample;
        if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
            printf("Error reading the line numbers\n");
            fclose(dataFile);
            MPI_Finalize();
            exit(1);
        }

        if (value > 0){
            if (sample < numTumor){
                sparseTumorData[gene].insert(sample);
				tumorSamples.insert(sample);
            }
            else{
                sparseNormalData[gene].insert(sample);
            }
        }
    }

    fclose(dataFile);
    return std::make_pair(sparseTumorData, sparseNormalData);
}

std::pair<double, std::array<int, 3>> maxF(std::vector<std::set<int>>& tumorData, std::vector<std::set<int>>& normalData, int Nt,  int Nn, double alpha, const std::vector<std::string>& local_lines) {
    double maxF = -1.0;
	std::array<int, 3> bestComb = {-1, -1, -1};
	for (const auto& line : local_lines) {
        std::istringstream iss(line);
        int i, j, k;
        iss >> i >> j >> k;
		
		const std::set<int>& gene1Tumor = tumorData[i];
        const std::set<int>& gene2Tumor = tumorData[j];
		const std::set<int>& gene3Tumor = tumorData[k];

		std::set<int> intersectTumor1;
		std::set<int> intersectTumor2;
        std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));
        std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));	
	
		
		const std::set<int>& gene1Normal = normalData[i];
        const std::set<int>& gene2Normal = normalData[j];
		const std::set<int>& gene3Normal = normalData[k];

		std::set<int> intersectNormal1;
		std::set<int> intersectNormal2;
        std::set_intersection(gene1Normal.begin(), gene1Normal.end(), gene2Normal.begin(), gene2Normal.end(), std::inserter(intersectNormal1, intersectNormal1.begin()));
        std::set_intersection(gene3Normal.begin(), gene3Normal.end(), intersectNormal1.begin(), intersectNormal1.end(), std::inserter(intersectNormal2, intersectNormal2.begin()));	


		int TP = intersectTumor2.size();
		int FP = intersectNormal2.size();

		int TN = Nn - FP;
		int FN = Nt - TP;

		double F = (alpha * TP + TN) / static_cast<double>(Nt + Nn);
		if (F >= maxF){
			maxF = F;
			bestComb = {i, j, k};
		}
		
    }
	
	return std::make_pair(maxF, bestComb);
}

int main(int argc, char** argv) {
	auto total_start = std::chrono::high_resolution_clock::now();
    int size, rank;
    initialize_mpi(argc, argv, size, rank);
	auto mpi_init_end = std::chrono::high_resolution_clock::now();

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <intermediate data file> <raw input file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
	auto read_data_start = std::chrono::high_resolution_clock::now();
	int numGenes, numSamples, numTumor, numNormal;
	std::set<int> tumorSamples;
    std::pair<std::vector<std::set<int>>, std::vector<std::set<int>>> dataPair = read_data(argv[2], numGenes, numSamples, numTumor, numNormal, tumorSamples);
    std::vector<std::set<int>>& tumorData = dataPair.first;
    std::vector<std::set<int>>& normalData = dataPair.second;	
	auto read_data_end = std::chrono::high_resolution_clock::now();

	auto get_line_count_start = std::chrono::high_resolution_clock::now();
    int line_count = get_line_count(argv[1], rank);
	auto get_line_count_end = std::chrono::high_resolution_clock::now();

	auto calc_line_indices_start = std::chrono::high_resolution_clock::now();
    int start_line, end_line;
    calculate_line_indices(rank, size, line_count, start_line, end_line);
	auto calc_line_indices_end = std::chrono::high_resolution_clock::now();	

	auto read_lines_segment_start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> local_lines = read_lines_segment(argv[1], start_line, end_line);
	auto read_lines_segment_end = std::chrono::high_resolution_clock::now();

	double alpha = 0.1;

	std::set<int> droppedSamples;
	int numComb = 0;	

	auto main_loop_start = std::chrono::high_resolution_clock::now();
	while(tumorSamples != droppedSamples){
			std::pair<double, std::array<int,3>> maxFOut = maxF(tumorData, normalData, numTumor, numNormal, alpha, local_lines);
			double localMaxF = maxFOut.first;
			std::array<int, 3> localBestComb = maxFOut.second;

			struct {
				double value;
				int rank;
			} localResult, globalResult;

			localResult.value = localMaxF;
			localResult.rank = rank;

			MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

			std::array<int, 3> globalBestComb;
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
				std::ofstream outfile("3hit_coverset_results.txt", std::ios::app);
				if (!outfile) {
					std::cerr << "Error: Could not open output file." << std::endl;
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
				outfile << numComb << "- (" << globalBestComb[0] << ", " << globalBestComb[1] << ", " << globalBestComb[2] << ")  F-max = " << globalResult.value << std::endl;
				outfile.close();
				numComb++;
			}
	}
	auto main_loop_end = std::chrono::high_resolution_clock::now();
	MPI_Finalize();
	auto mpi_finalize_end = std::chrono::high_resolution_clock::now(); 

    auto total_end = std::chrono::high_resolution_clock::now(); 

    std::chrono::duration<double> total_time = total_end - total_start;
    std::chrono::duration<double> read_data_time = read_data_end - read_data_start;
    std::chrono::duration<double> get_line_count_time = get_line_count_end - get_line_count_start;
    std::chrono::duration<double> calc_line_indices_time = calc_line_indices_end - calc_line_indices_start;
    std::chrono::duration<double> read_lines_segment_time = read_lines_segment_end - read_lines_segment_start;
    std::chrono::duration<double> main_loop_time = main_loop_end - main_loop_start;

    // Only rank 0 writes to output.txt
    if (rank == 0) {
        std::ofstream outfile("output.txt", std::ios::app);
        if (!outfile) {
            std::cerr << "Error: Could not open output file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outfile << "----------------------Timing Information for 3 hit set cover----------------------" << std::endl;
        outfile << "Total time: " << total_time.count() << " seconds" << std::endl;
        outfile << "Time for reading data: " << read_data_time.count() << " seconds" << std::endl;
        outfile << "Time for getting line count: " << get_line_count_time.count() << " seconds" << std::endl;
        outfile << "Time for calculating line indices: " << calc_line_indices_time.count() << " seconds" << std::endl;
        outfile << "Time for reading lines segment: " << read_lines_segment_time.count() << " seconds" << std::endl;
        outfile << "Time for main loop: " << main_loop_time.count() << " seconds" << std::endl;
        outfile << std::endl;
        outfile.close();
    }
    return 0;
}

