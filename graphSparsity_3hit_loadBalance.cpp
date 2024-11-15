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
#include <queue>    
#include <utility>

#define MAX_BUF_SIZE 1024
#define CHUNK_SIZE 2LL

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

void funcName(std::vector<Gene> data, long long int startComb, long long int endComb, int totalGenes, long long int &count, int rank, int size){
	
	
	for (long long int lambda = startComb; lambda < endComb; lambda++){
		printf("Here is lambda: %lld, startComb: %lld, endComb: %lld\n", lambda, startComb, endComb);
		long long int j = static_cast<long long int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
		long long int i = lambda - (j * (j - 1)) / 2;

		Gene gene1 = data[i];
		Gene gene2 = data[j];

		std::set<int> intersectNormal1, intersectTumor1;

		std::set_intersection(gene1.tumor.begin(), gene1.tumor.end(), gene2.tumor.begin(), gene2.tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));
		std::set_intersection(gene1.normal.begin(), gene1.normal.end(), gene2.normal.begin(), gene2.normal.end(), std::inserter(intersectNormal1, intersectNormal1.begin()));
		
		if (!intersectTumor1.empty()) {	

			for (long long int k = j + 1; k < totalGenes; k++){
					Gene gene3 = data[k];
					std::set<int> intersectNormal2, intersectTumor2;
							std::set_intersection(gene3.tumor.begin(), gene3.tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));
					std::set_intersection(gene3.normal.begin(), gene3.normal.end(), intersectNormal1.begin(), intersectNormal1.end(), std::inserter(intersectNormal2, intersectNormal2.begin()));
					
					if (!intersectTumor2.empty()){
						count++;
					}
				}

		}

	}

}


int main(int argc, char *argv[]){
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	const int WORK_REQUEST_TAG = 1;
	const int WORK_ASSIGN_TAG = 2;
	const int STEAL_TAG = 3;

	double total_start_time = MPI_Wtime();
	if (argc != 2){
		printf("One argument expected: ./graphSparcity <dataFile>");
		MPI_Finalize();
		return 1;
	}
	double start_time, end_time;
    	double elapsed_time_loading, elapsed_time_func, elapsed_time_total;	
	start_time = MPI_Wtime();
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

	std::vector<Gene> sparseData (numGenes);

	int fileRows = numGenes * numSamples;
	
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
	end_time = MPI_Wtime();
	elapsed_time_loading = end_time - start_time;

	start_time = MPI_Wtime();

	int num_workers = size - 1;
	
	long long int num_Comb, num_startComb, endComb, chunkSize, remainder;
	num_Comb = nCr(numGenes, 2);
	chunkSize = num_Comb / num_workers;
	remainder = num_Comb % num_workers;
	long long int count = 0;

	if (rank != 0){

		long long int startComb = (rank-1) * chunkSize + ((rank-1) < remainder ? (rank-1) : remainder);
		long long int endComb = startComb + chunkSize + ((rank-1) < remainder ? 1 : 0);
		
		bool hasWork = true;
		while(hasWork){
			for (long long int i = startComb; i < endComb; i += CHUNK_SIZE) {
			    long long int currentStartComb = i;
			    long long int currentEndComb = std::min(i + CHUNK_SIZE, endComb);
			    funcName(sparseData, currentStartComb, currentEndComb, numGenes, count, rank, size);
			    
		            MPI_Status status;
			    int flag;
			    MPI_Iprobe(0, STEAL_TAG, MPI_COMM_WORLD, &flag, &status);
			    if (flag) {
				long long int stealEnd;
				MPI_Recv(&stealEnd, 1, MPI_LONG_LONG_INT, 0, STEAL_TAG, MPI_COMM_WORLD, &status);
				// Adjust endComb to give up work to master
				endComb = stealEnd;		
		            }
			
			}
		
			// Notify master that worker is ready for more work
			MPI_Send(&rank, 1, MPI_INT, 0, WORK_REQUEST_TAG, MPI_COMM_WORLD);

			// Receive new work assignment from master
			MPI_Status status;
			MPI_Recv(&startComb, 1, MPI_LONG_LONG_INT, 0, WORK_ASSIGN_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&endComb, 1, MPI_LONG_LONG_INT, 0, WORK_ASSIGN_TAG, MPI_COMM_WORLD, &status);

			// Check if no more work is available
			if (startComb >= endComb) {
			    hasWork = false;
			}
		}
	}
	else{
		std::queue<std::pair<long long int, long long int> > workQueue;
		
		long long int assignedCombs = 0;
		
		for (int i = 1; i < size; i++){
		
			long long int workerChunkSize = chunkSize + (i < (num_Comb % num_workers) ? 1 : 0);
			assignedCombs += workerChunkSize;
		}

		if (assignedCombs < num_Comb) {
     			workQueue.push({assignedCombs, num_Comb});
    		}

		int workersDone = 0;
		bool workAvailable = !workQueue.empty();
		int workerRank;
		while (workersDone < num_workers){

			MPI_Status status;
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &flag, &status);
        		
			if (flag) {
				int workerRank;
            			MPI_Recv(&workerRank, 1, MPI_INT, MPI_ANY_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
				
				if (workAvailable) {
					std::pair<long long int, long long int> workRange = workQueue.front();
					long long int newStart = workRange.first;
					long long int newEnd = workRange.second;
					workQueue.pop();
					if (workQueue.empty()) {
					    workAvailable = false;
					}
					MPI_Send(&newStart, 1, MPI_LONG_LONG_INT, workerRank, WORK_ASSIGN_TAG, MPI_COMM_WORLD);
					MPI_Send(&newEnd, 1, MPI_LONG_LONG_INT, workerRank, WORK_ASSIGN_TAG, MPI_COMM_WORLD);
				}
				else{

					long long int noWork = 0;
					MPI_Send(&noWork, 1, MPI_LONG_LONG_INT, workerRank, WORK_ASSIGN_TAG, MPI_COMM_WORLD);
					MPI_Send(&noWork, 1, MPI_LONG_LONG_INT, workerRank, WORK_ASSIGN_TAG, MPI_COMM_WORLD);
					workersDone++;

				}
			}
		}
	}

	
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

	MPI_Finalize();
	return 0;
}
