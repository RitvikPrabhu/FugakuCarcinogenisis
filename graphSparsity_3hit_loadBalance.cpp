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
#define CHUNK_SIZE 100000LL

struct Gene {
	std::set<int> tumor;
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

void process_lambda_interval(std::vector<Gene> data, long long int startComb, long long int endComb, int totalGenes, long long int &count, int rank, int size){
	
	
	for (long long int lambda = startComb; lambda < endComb; lambda++){
		long long int j = static_cast<long long int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
		long long int i = lambda - (j * (j - 1)) / 2;

		Gene gene1 = data[i];
		Gene gene2 = data[j];

		std::set<int> intersectTumor1;

		std::set_intersection(gene1.tumor.begin(), gene1.tumor.end(), gene2.tumor.begin(), gene2.tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));
		
		if (!intersectTumor1.empty()) {	

			for (long long int k = j + 1; k < totalGenes; k++){
					Gene gene3 = data[k];
					std::set<int> intersectTumor2;
							std::set_intersection(gene3.tumor.begin(), gene3.tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));
					
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
	

	const int WORK_DONE_TAG = 1;
	const int WORK_RECEIVE_TAG = 2;
	const int WORK_TERMINATION_TAG = 3; 
	double total_start_time = MPI_Wtime();
	if (argc != 2){
		printf("One argument expected: ./graphSparcity <dataFile>");
		MPI_Finalize();
		return 1;
	}
	double start_time, end_time;
    	double elapsed_time_loading, elapsed_time_func, elapsed_time_total;	
	start_time = MPI_Wtime();
	//Read Data function start
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
		}
	}
	
	fclose(dataFile);
	//Read data function should end and returns the sparse data
	end_time = MPI_Wtime();
	elapsed_time_loading = end_time - start_time;

	start_time = MPI_Wtime();

//////////////////////////////////////////////////////////////////////////////////////////////
	int num_workers = size - 1;
	long long int num_Comb, num_startComb, endComb, remainder;
	num_Comb = nCr(numGenes, 2);
	remainder = num_Comb % num_workers;
	long long int count = 0;
	if (rank == 0){ //Master
		int next_idx = num_workers * CHUNK_SIZE;
		while (next_idx < num_Comb){
			MPI_Status status;
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
			
			if (flag == 1){
				char c;
				int workerRank = status.MPI_SOURCE;
				MPI_Recv(&c,1,MPI_CHAR,workerRank,WORK_DONE_TAG,MPI_COMM_WORLD,&status);
				if (c == 'a'){
					MPI_Send(&next_idx,1,MPI_INT,workerRank,WORK_RECEIVE_TAG,MPI_COMM_WORLD);
					next_idx += CHUNK_SIZE; 
				}
			}
		}
		int term_signal = -1;
		for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
			int term_signal = -1;  // Using -1 as the termination signal
			MPI_Send(&term_signal, 1, MPI_INT, workerRank, WORK_RECEIVE_TAG, MPI_COMM_WORLD);
		}
	}
	else{ //Worker
		int begin = (rank-1) * CHUNK_SIZE;
		int end = std::min(begin + CHUNK_SIZE, num_Comb);
		MPI_Status status;
		while (end <= num_Comb){
			process_lambda_interval(sparseData, begin, end, numGenes, count, rank, size);
			MPI_Request request;
			char c = 'a';
			MPI_Isend(&c,1,MPI_CHAR,0,WORK_DONE_TAG,MPI_COMM_WORLD, &request);
			int next_idx;
			MPI_Recv(&next_idx,1,MPI_INT,0,WORK_RECEIVE_TAG,MPI_COMM_WORLD,&status);
			begin = next_idx;
			if (begin == -1) break;
			end = std::min(next_idx + CHUNK_SIZE, num_Comb);
		}
	
	}


//////////////////////////////////////////////////////////////////////////////////////////////
	
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
//Put the below code in a function	
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
