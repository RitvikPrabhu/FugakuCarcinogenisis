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

void process_lambda_interval(const std::vector<std::set<int>>& tumorData, const std::vector<std::set<int>>& normalData, long long int startComb, long long int endComb, int totalGenes, long long int &count, std::array<int, 4>& bestCombination, const std::vector<std::array<int32_t, 3>>& workload, int Nt, int Nn, double& maxF){
    double alpha = 1;
    for (long long int lambda = startComb; lambda <= endComb; lambda++){
		
	if (lambda == 0) continue;

	double q = std::pow(std::sqrt(729.0 * lambda * lambda - 3.0) + 27.0 * lambda, 1.0 / 3.0);

        int k = static_cast<int>(std::floor(std::pow(q / 9.0, 1.0 / 3.0) + 1.0 / std::pow(3.0 * q, 1.0 / 3.0) - 1.0));

        int Tz = (k * (k + 1) * (k + 2)) / 6;
        int lambda_prime = lambda - Tz;

        int j = static_cast<int>(std::floor(std::sqrt(0.25 + 2.0 * lambda_prime) - 0.5));

        int i = lambda_prime - (j * (j + 1)) / 2;

	const std::set<int>& gene1Tumor = tumorData[i];
        const std::set<int>& gene2Tumor = tumorData[j];

        std::set<int> intersectTumor1;
        std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));

        if (!intersectTumor1.empty()) {
			const std::set<int>& gene3Tumor = tumorData[k];
			std::set<int> intersectTumor2;
			std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));

			if (!intersectTumor2.empty()){
				
				for (int l = k + 1; l < totalGenes; l++){

					const std::set<int>& gene4Tumor = tumorData[l];
					std::set<int> intersectTumor3;
					std::set_intersection(gene4Tumor.begin(), gene4Tumor.end(), intersectTumor2.begin(), intersectTumor2.end(), std::inserter(intersectTumor3, intersectTumor3.begin()));
					if (!intersectTumor3.empty()){
						                                                const std::set<int>& gene1Normal = normalData[i];
                                        const std::set<int>& gene2Normal = normalData[j];
                                                const std::set<int>& gene3Normal = normalData[k];
                                        const std::set<int>& gene4Normal = normalData[l];

                                        std::set<int> intersectNormal1;
                                        std::set<int> intersectNormal2;
                                        std::set<int> intersectNormal3;


                                        std::set_intersection(gene1Normal.begin(), gene1Normal.end(), gene2Normal.begin(), gene2Normal.end(), std::inserter(intersectNormal1, intersectNormal1.begin()));
                                                std::set_intersection(gene3Normal.begin(), gene3Normal.end(), intersectNormal1.begin(), intersectNormal1.end(), std::inserter(intersectNormal2, intersectNormal2.begin()));
                                                std::set_intersection(gene4Normal.begin(), gene4Normal.end(), intersectNormal2.begin(), intersectNormal2.end(), std::inserter(intersectNormal3, intersectNormal3.begin()));

                                                int TP = intersectTumor3.size();
                                                int TN = intersectNormal3.size();

                                                double F = (alpha * TP + TN) / static_cast<double>(Nt + Nn);

                                                if (F >= maxF){
                                                        maxF = F;
                                                        bestCombination = {i, j, k, l};

                                                }				

					}

				}
			}
        }
    }
}

void write_timings_to_file(const double all_times[][3], int size, long long int totalCount, const char* filename) {
    //std::ofstream timingFile(filename);
    std::ostream& timingFile = std::cout;
    //if (timingFile.is_open()) {
		timingFile << "----------------------Timing Information for 4-hit sparsification----------------------" << std::endl;
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
		fflush(stdout);
       // timingFile.close();
    //} else {
    //    printf("Error opening timings output file\n");
    //}
}

std::string* read_data(const char* filename, int& numGenes, int& numSamples, int& numTumor, int& numNormal, std::set<int>& tumorSamples, std::vector<std::set<int>>& sparseTumorData, std::vector<std::set<int>>& sparseNormalData) {
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

        sparseTumorData.resize(numGenes);
    sparseNormalData.resize(numGenes);

    int fileRows = numGenes * numSamples;
    std::string* geneIdArray = new std::string[numGenes];

    for (int i = 0; i < fileRows; i++){
        int gene, sample;
        if (fscanf(dataFile, "%d %d %d %s %s\n", &gene, &sample, &value, geneId, sampleId) != 5) {
            printf("Error reading the line numbers\n");
            fclose(dataFile);
            MPI_Finalize();
            exit(1);
        }

                geneIdArray[gene] = geneId;

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
    return geneIdArray;
}

void master_process(int num_workers, long long int num_Comb) {
    long long int next_idx = num_workers * CHUNK_SIZE;
    while (next_idx < num_Comb) {
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag == 1) {
            char c;
            int workerRank = status.MPI_SOURCE;
            MPI_Recv(&c, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);
            if (c == 'a') {
                MPI_Send(&next_idx, 1, MPI_LONG_LONG_INT, workerRank, 2, MPI_COMM_WORLD);
                next_idx += CHUNK_SIZE;
            }
        }
    }
    for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
        long long int term_signal = -1;
        MPI_Send(&term_signal, 1, MPI_LONG_LONG_INT, workerRank, 2, MPI_COMM_WORLD);
    }
}

void worker_process(int rank, long long int num_Comb,
                    std::vector<std::set<int>>& tumorData,
                    const std::vector<std::set<int>>& normalData,
                    int numGenes, long long int& count, int Nt, int Nn, const char* hit3_file, double& localBestMaxF, std::array<int, 4>& localComb) {

                        long long int begin = (rank - 1) * CHUNK_SIZE;
                        long long int end = std::min(begin + CHUNK_SIZE, num_Comb);
                        MPI_Status status;
                        while (end <= num_Comb) {
                                std::vector<std::array<int32_t, 3>> workload = read_triplets_segment(hit3_file, begin, end);
                                process_lambda_interval(tumorData, normalData, begin, end, numGenes, count, localComb, workload, Nt, Nn, localBestMaxF);
                                char c = 'a';
                                MPI_Send(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

                                long long int next_idx;
                                MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);
                                if (next_idx == -1) break;

                                begin = next_idx;
                                end = std::min(next_idx + CHUNK_SIZE, num_Comb);
                        }

}



void distribute_tasks(int rank, int size, int numGenes,
                      std::vector<std::set<int>>& tumorData,
                      const std::vector<std::set<int>>& normalData, long long int& count,
                      int Nt, int Nn, const char* outFilename, const char* hit3_file, const std::set<int>& tumorSamples, std::string* geneIdArray) {


        long long int num_Comb = get_triplet_count(hit3_file, rank);
        std::set<int> droppedSamples;
        while(tumorSamples != droppedSamples){
                        std::array<int, 4> localComb = {-1, -1, -1, -1};
                        double localBestMaxF = -1.0;
                        if (rank == 0) { // Master
                                master_process(size - 1, num_Comb);
                        } else { // Worker
                                worker_process(rank, num_Comb, tumorData, normalData,
                                                                                         numGenes, count, Nt, Nn, hit3_file, localBestMaxF, localComb);
                        }

                struct {
            double value;
            int rank;
        } localResult, globalResult;

        localResult.value = localBestMaxF;
        localResult.rank = rank;

                MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

                std::array<int, 4> globalBestComb;
        if (rank == globalResult.rank) {
            globalBestComb = localComb;
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
	//Nt -= sampleToCover.size();

                if (rank == 0) {
		    std::ofstream outfile(outFilename, std::ios::app);
		    if (!outfile) {
			std::cerr << "Error: Could not open output file." << std::endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		    }
		    outfile << "(";
		    for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
			outfile << geneIdArray[globalBestComb[idx]];
			if (idx != globalBestComb.size() - 1) {
			    outfile << ", ";
			}
		    }
		    outfile << ")  F-max = " << globalResult.value << std::endl;
		    outfile.close();
		}
        }

}



int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4){
        printf("Three argument expected: ./graphSparcity <dataFile> <outputMetricFile> <prunedDataOutputFile>");
        MPI_Finalize();
        return 1;
    }
    
    double total_start_time = MPI_Wtime();



    double start_time, end_time;
    double elapsed_time_loading, elapsed_time_func, elapsed_time_total;


    start_time = MPI_Wtime();
    int numGenes, numSamples, numTumor, numNormal;
    std::set<int> tumorSamples;
    std::vector<std::set<int>> tumorData;
    std::vector<std::set<int>> normalData;
    std::string* geneIdArray = read_data(argv[1], numGenes, numSamples, numTumor, numNormal, tumorSamples, tumorData, normalData);
    end_time = MPI_Wtime();
    elapsed_time_loading = end_time - start_time;



    start_time = MPI_Wtime();
    long long int totalCount = 0;
    distribute_tasks(rank, size, numGenes, tumorData, normalData, totalCount, numTumor, numNormal, argv[3], argv[2], tumorSamples, geneIdArray);
    double total_end_time = MPI_Wtime();
    elapsed_time_total = total_end_time - total_start_time;
    end_time = MPI_Wtime();
    elapsed_time_func = end_time - start_time;



    double elapsed_times[3] = {elapsed_time_loading, elapsed_time_func, elapsed_time_total};
    double all_times[size][3];
    MPI_Gather(elapsed_times, 3, MPI_DOUBLE, all_times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    if (rank == 0) {
        write_timings_to_file(all_times, size, totalCount, argv[2]);
    }

    MPI_Finalize();
    return 0;
}

