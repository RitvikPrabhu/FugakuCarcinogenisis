#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <tuple>

#define MAX_BUF_SIZE 1024

struct Gene {
    std::set<int> tumor;
    std::set<int> normal;
};

long long int nCr(int n, int r) {
    if (r > n) return 0;
    if (r == 0 || r == n) return 1;
    if (r > n - r) r = n - r;  // Because C(n, r) == C(n, n-r)

    long long int result = 1;
    for (int i = 1; i <= r; ++i) {
        result *= (n - r + i);
        result /= i;
    }
    return result;
}

void funcName(std::vector<Gene> data, int totalGenes, int numTumor, int numNormal, int rank, int size, std::vector<int> &gene1_idx, std::vector<int> &gene2_idx, std::vector<int> &tumorCounts){
    long long int num_Comb, startComb, endComb, chunkSize, remainder;
    num_Comb = nCr(totalGenes, 2);    
    chunkSize = num_Comb / size;
    remainder = num_Comb % size;

    startComb = rank * chunkSize + (rank < remainder ? rank : remainder);
    endComb = startComb + chunkSize + (rank < remainder ? 1 : 0);
    
    for (long long int lambda = startComb; lambda < endComb; lambda++){
        long long int j = static_cast<long long int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
        long long int i = lambda - (j * (j - 1)) / 2;

        Gene gene1 = data[i];
        Gene gene2 = data[j];

        std::set<int> intersectTumor1;
        std::set_intersection(gene1.tumor.begin(), gene1.tumor.end(), gene2.tumor.begin(), gene2.tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));
        
        if (!intersectTumor1.empty()) {    
            gene1_idx.push_back(static_cast<int>(i));
            gene2_idx.push_back(static_cast<int>(j));
            tumorCounts.push_back(static_cast<int>(intersectTumor1.size()));
        }
    }
}

void processInnerLoop(const std::vector<int> &gene1_idx, const std::vector<int> &gene2_idx, const std::vector<int> &tumorCounts, const std::vector<Gene> &sparseData, int numGenes, long long int &count) {
    for (size_t idx = 0; idx < gene1_idx.size(); idx++) {
        int i = gene1_idx[idx];
        int j = gene2_idx[idx];
        int tumorCount = tumorCounts[idx];

        for (int k = j + 1; k < numGenes; k++) {
            Gene gene3 = sparseData[k];
            std::set<int> intersectTumor2;
            std::set_intersection(gene3.tumor.begin(), gene3.tumor.end(), sparseData[j].tumor.begin(), sparseData[j].tumor.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));
            if (!intersectTumor2.empty()) {
                count++;
            }
        }
    }
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2){
        printf("One argument expected: ./graphSparcity <dataFile>\n");
        MPI_Finalize();
        return 1;
    }
    double start_time, end_time;
    double elapsed_time_loading, elapsed_time_func;    
    start_time = MPI_Wtime(); // Start timing the data loading stage
    FILE* dataFile;
    dataFile = fopen(argv[1], "r");
    
    if (dataFile == NULL) {
        perror("Error opening file");
        MPI_Finalize();
        return 1; 
    }

    char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];
    int numGenes, numSamples, value, numTumor, numNormal;
    if (fscanf(dataFile, "%d %d %d %d %d\n", &numGenes, &numSamples, &value, &numTumor, &numNormal) != 5) {
        printf("Error reading the first line numbers\n");
        fclose(dataFile);
        MPI_Finalize();
        return 1; 
    }

    // Create a data structure to store information about each gene
    std::vector<Gene> sparseData (numGenes);
    int fileRows = numGenes * numSamples;
    
    // Load data row by row and categorize samples as either tumor or normal
    for (int i = 0; i < fileRows; i++){
        int gene, sample;    
        if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
            printf("Error reading the line numbers\n");
            fclose(dataFile);
            MPI_Finalize();
            return 1; 
        }
    
        if (value > 0){
            if (sample < numTumor){
                sparseData[gene].tumor.insert(sample);
            }
            else{
                sparseData[gene].normal.insert(sample);
            }
        }
    }
    
    fclose(dataFile);
    end_time = MPI_Wtime(); // End timing the data loading stage
    elapsed_time_loading = end_time - start_time;
    
    // Prepare data structures for holding pairs of genes with common tumor samples
    std::vector<int> localGene1Idx;
    std::vector<int> localGene2Idx;
    std::vector<int> localTumorCounts;
    start_time = MPI_Wtime(); // Start timing the computation stage

    // Each process calls funcName to find intersecting tumor samples for its assigned gene pairs
    funcName(sparseData, numGenes, numTumor, numNormal, rank, size, localGene1Idx, localGene2Idx, localTumorCounts);
    
    int localTractableSize = localGene1Idx.size();
    std::vector<int> rankDataSizes(size);

    // Gather the sizes of results from all processes at rank 0
    MPI_Gather(&localTractableSize, 1, MPI_INT, rankDataSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate where each rank’s data will go in the global array on rank 0
    std::vector<int> rank_displacement_idx(size);
    int totalWorkSize = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            rank_displacement_idx[i] = totalWorkSize;
            totalWorkSize += rankDataSizes[i];
        }
    }

    // Create global arrays to store data from all processes (only on rank 0)
    std::vector<int> globalGene1Idx;
    std::vector<int> globalGene2Idx;
    std::vector<int> globalTumorCounts;
    if (rank == 0) {
        globalGene1Idx.resize(totalWorkSize);
        globalGene2Idx.resize(totalWorkSize);
        globalTumorCounts.resize(totalWorkSize);
    }

    // Gather results from all ranks into global arrays on rank 0
    MPI_Gatherv(localGene1Idx.data(), localTractableSize, MPI_INT, 
                globalGene1Idx.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localGene2Idx.data(), localTractableSize, MPI_INT, 
                globalGene2Idx.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localTumorCounts.data(), localTractableSize, MPI_INT, 
                globalTumorCounts.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> redistributedGene1Idx;
    std::vector<int> redistributedGene2Idx;
    std::vector<int> redistributedTumorCounts;
    if (rank == 0) {
        int newChunkSize = totalWorkSize / size; //Basic size of work per rank
        int remainder = totalWorkSize % size; //In case the amount of data is not divisble

        rank_displacement_idx[0] = 0; //Rank 0 has to start at index 0 on the global array
	// The below for loop dtermines how many items each rank should get and calculates start positions
        for (int i = 0; i < size; i++) {
            rankDataSizes[i] = newChunkSize + (i < remainder ? 1 : 0);
            
	    // Calculate the starting index for each rank’s portion in the global array
	    if (i > 0) {
                rank_displacement_idx[i] = rank_displacement_idx[i - 1] + rankDataSizes[i - 1];
            }
        }
    }

    // Scatter redistributed data back to each process
    redistributedGene1Idx.resize(rankDataSizes[rank]);
    redistributedGene2Idx.resize(rankDataSizes[rank]);
    redistributedTumorCounts.resize(rankDataSizes[rank]);
    MPI_Scatterv(globalGene1Idx.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 
                 redistributedGene1Idx.data(), rankDataSizes[rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalGene2Idx.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 
                 redistributedGene2Idx.data(), rankDataSizes[rank], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalTumorCounts.data(), rankDataSizes.data(), rank_displacement_idx.data(), MPI_INT, 
                 redistributedTumorCounts.data(), rankDataSizes[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Count the number of three-way intersections (basically the inner loop of the lambda program)
    long long int count = 0;
    processInnerLoop(redistributedGene1Idx, redistributedGene2Idx, redistributedTumorCounts, sparseData, numGenes, count);

    end_time = MPI_Wtime(); //Ending of Computation time
    elapsed_time_func = end_time - start_time;

   
    // Combine results from all ranks to get the total count of three-way intersections
    long long int totalCount = 0;
    MPI_Reduce(&count, &totalCount, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    //Record Computation time
    double elapsed_times[2] = {elapsed_time_loading, elapsed_time_func};
    double all_times[2][size];

    // Gather time metrics from all ranks to rank 0
    MPI_Gather(elapsed_times, 2, MPI_DOUBLE, all_times, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 outputs the timing data and total count to a file
     if (rank == 0) {
        std::ofstream timingFile("output.txt");
        if (timingFile.is_open()) {
            for (int stage = 0; stage < 2; ++stage) {
                double max_time = -1.0, min_time = 1e9, total_time = 0.0;
                int rank_max = 0, rank_min = 0;
                for (int i = 0; i < size; ++i) {
                    double time = all_times[stage][i];
                    if (time > max_time) {
                        max_time = time;
                        rank_max = i;
                    }
                    if (time < min_time) {
                        min_time = time;
                        rank_min = i;
                    }
                    total_time += time;
                }
                double avg_time = total_time / size;
                if (stage == 0) {
                    timingFile << "Stage " << stage << " (Loading Data):\n";
                }
                else if (stage == 1) {
                    timingFile << "Stage " << stage << " (Computation):\n";
                }
                timingFile << "Rank " << rank_max << " took the longest time: " << max_time << " seconds.\n";
                timingFile << "Rank " << rank_min << " took the shortest time: " << min_time << " seconds.\n";
                timingFile << "Average time: " << avg_time << " seconds.\n\n";
            }
            timingFile << "Total number of combinations: " << totalCount << "\n";
            timingFile.close();
        } else {
            printf("Error opening timings output file\n");
        }
    }

    MPI_Finalize();
    return 0;
}

