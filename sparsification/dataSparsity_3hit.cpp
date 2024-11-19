#include <utility>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>

#define MAX_BUF_SIZE 1024
#define CHUNK_SIZE 100000LL

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

void process_lambda_interval(const std::vector<std::set<int>>& tumorData, long long int startComb, long long int endComb, int totalGenes, long long int &count, int rank, std::vector<std::array<int, 3>>& bestCombinations){
    for (long long int lambda = startComb; lambda < endComb; lambda++){
        int j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
        int i = lambda - (j * (j - 1)) / 2;

        const std::set<int>& gene1Tumor = tumorData[i];
        const std::set<int>& gene2Tumor = tumorData[j];

        std::set<int> intersectTumor1;
        std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));

        if (!intersectTumor1.empty()) {
            for (int k = j + 1; k < totalGenes; k++){
                const std::set<int>& gene3Tumor = tumorData[k];
                std::set<int> intersectTumor2;
                std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));

                if (!intersectTumor2.empty()){
					bestCombinations.push_back({i, j, k});
                    count++;
                }
            }
        }
    }
}

void write_timings_to_file(const double all_times[][3], int size, long long int totalCount) {
    std::ofstream timingFile("output.txt");
    if (timingFile.is_open()) {
        for (int stage = 0; stage < 3; ++stage) {
            double max_time = -1.0, min_time = 1e9, total_time = 0.0;
            int rank_max = 0, rank_min = 0;
            for (int i = 0; i < size; ++i) {
                double time = all_times[i][stage];
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
            } else if (stage == 1) {
                timingFile << "Stage " << stage << " (Computation):\n";
            } else {
                timingFile << "Stage " << stage << " (Total Time):\n";
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

std::pair<std::vector<std::set<int>>, std::vector<std::set<int>>> read_data(const char* filename, int& numGenes, int& numSamples, int& numTumor, int& numNormal) {
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
            } 
			else{
				sparseNormalData[gene].insert(sample);
			}
        }
    }

    fclose(dataFile);
    return std::make_pair(sparseTumorData, sparseNormalData);;
}

void master_process(int num_workers, long long int num_Comb) {
    int next_idx = num_workers * CHUNK_SIZE;
    while (next_idx < num_Comb) {
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag == 1) {
            char c;
            int workerRank = status.MPI_SOURCE;
            MPI_Recv(&c, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);
            if (c == 'a') {
                MPI_Send(&next_idx, 1, MPI_INT, workerRank, 2, MPI_COMM_WORLD);
                next_idx += CHUNK_SIZE;
            }
        }
    }
    for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
        int term_signal = -1;
        MPI_Send(&term_signal, 1, MPI_INT, workerRank, 2, MPI_COMM_WORLD);
    }
}

void worker_process(int rank, int num_workers, long long int num_Comb, const std::vector<std::set<int>>& tumorData, const std::vector<std::set<int>>& normalData, int numGenes, long long int& count, int Nt, int Nn) {
    int begin = (rank - 1) * CHUNK_SIZE;
    int end = std::min(begin + CHUNK_SIZE, num_Comb);
    MPI_Status status;
	std::vector<std::array<int, 3>> localComb;
    while (end <= num_Comb) {
        process_lambda_interval(tumorData, begin, end, numGenes, count, rank, localComb);
        MPI_Request request;
        char c = 'a';
        MPI_Isend(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &request);
        int next_idx;
        MPI_Recv(&next_idx, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        begin = next_idx;
        if (begin == -1) break;
        end = std::min(next_idx + CHUNK_SIZE, num_Comb);
    }
/*	
	// Calculate the offset based on ranks
	long long int prev_offset;
	
	if (rank == 1) {
			prev_offset = 0;
	}
	else {
		MPI_Recv(&prev_offset, 1, MPI_LONG_LONG, rank - 1, 0, MPI_COMM_WORLD, &status);
	}

	// Prepare data for writing
	std::string localData;
	for (const auto& comb : localComb) {
		localData += std::to_string(comb[0]) + " " + std::to_string(comb[1]) + " " + std::to_string(comb[2]) + "\n";
	}
	long long int local_size = static_cast<long long int>(localData.size());
	MPI_Offset file_offset = prev_offset;

	// Send offset to the next rank
	if (rank < num_workers) {
		long long int next_offset = file_offset + local_size;
		MPI_Send(&next_offset, 1, MPI_LONG_LONG, rank + 1, 0, MPI_COMM_WORLD);
	}


	MPI_File file;
	MPI_File_open(MPI_COMM_WORLD, "prunedData.txt", 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &file);

    // Write data at the calculated offset
    MPI_File_write_at(file, file_offset, 
                      localData.c_str(), 
                      static_cast<int>(localData.size()), 
                      MPI_CHAR, MPI_STATUS_IGNORE);

    // Close the file
    MPI_File_close(&file);*/
}

void distribute_tasks(int rank, int size, int numGenes, const std::vector<std::set<int>>& tumorData, const std::vector<std::set<int>>& normalData, long long int& count, int Nt, int Nn) {
    int num_workers = size - 1;
    long long int num_Comb, remainder;
    num_Comb = nCr(numGenes, 2);
    remainder = num_Comb % num_workers;

    if (rank == 0) { // Master
        master_process(num_workers, num_Comb);
    } else { // Worker
        worker_process(rank, num_workers, num_Comb, tumorData, normalData, numGenes, count, Nt, Nn);
    }
}

int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2){
        printf("One argument expected: ./graphSparcity <dataFile>");
        MPI_Finalize();
        return 1;
    }
    
    double total_start_time = MPI_Wtime();



    double start_time, end_time;
    double elapsed_time_loading, elapsed_time_func, elapsed_time_total;


    start_time = MPI_Wtime();
    int numGenes, numSamples, numTumor, numNormal;
    std::pair<std::vector<std::set<int>>, std::vector<std::set<int>>> dataPair = read_data(argv[1], numGenes, numSamples, numTumor, numNormal);
	std::vector<std::set<int>>& tumorData = dataPair.first;   
    std::vector<std::set<int>>& normalData = dataPair.second;
    end_time = MPI_Wtime();
    elapsed_time_loading = end_time - start_time;



    start_time = MPI_Wtime();
    long long int count = 0;
    distribute_tasks(rank, size, numGenes, tumorData, normalData, count, numTumor, numNormal);
    double total_end_time = MPI_Wtime();
    elapsed_time_total = total_end_time - total_start_time;
    end_time = MPI_Wtime();
    elapsed_time_func = end_time - start_time;



    double elapsed_times[3] = {elapsed_time_loading, elapsed_time_func, elapsed_time_total};
    double all_times[size][3];
    MPI_Gather(elapsed_times, 3, MPI_DOUBLE, all_times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    long long int totalCount = 0;
    MPI_Reduce(&count, &totalCount, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        write_timings_to_file(all_times, size, totalCount);
    }

	printf("Num Genes: %d, Num Samples: %d, num Tumor: %d, num Normal: %d\n", numGenes, numSamples, numTumor, numNormal);

    MPI_Finalize();
    return 0;
}

