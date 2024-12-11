#include <utility>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <limits>
#include <chrono>
#include <ctime> 
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

void process_lambda_interval(const std::vector<std::set<int>>& tumorData, const std::vector<std::set<int>>& normalData, long long int startComb, long long int endComb, int totalGenes, long long int &count, std::array<int, 4>& bestCombination, int Nt, int Nn, double& maxF){
		double alpha = 0.1;
		for (long long int lambda = startComb; lambda <= endComb; lambda++){

				if (lambda <= 0) continue; // Avoid division by zero and negative values

				double term1 = 243.0 * lambda - 1.0 / lambda;

				double rhs = (log(3.0 * lambda) + log(term1)) / 2.0;
				double A = exp(rhs);

				double common_numerator = pow(A + 27.0 * lambda, 1.0 / 3.0);
				double common_denominator = pow(3.0, 2.0 / 3.0);

				double v = (common_numerator / common_denominator) +
						(1.0 / (common_numerator * pow(3.0, 1.0 / 3.0))) - 1.0;

				unsigned long long int k_long = static_cast<unsigned long long int>(v);
				unsigned long long int Tz = k_long * (k_long + 1) * (k_long + 2) / 6;

				unsigned long long int LambdaP = lambda - Tz;

				int k = static_cast<int>(k_long);
				int j = static_cast<int>(sqrt(0.25 + 2.0 * LambdaP) - 0.5);

				unsigned long long int T2Dy = j * (j + 1) / 2;

				int i = static_cast<int>(LambdaP - T2Dy);
				if (i >= j || j >=k || i >= k) continue;

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

										if (!intersectTumor3.empty()){

												int TP = intersectTumor3.size();
												int TN = Nn - intersectNormal3.size();

												double F = (alpha * TP + TN) / static_cast<double>(Nt + Nn);
												//double F = static_cast<double>(alpha * TP + TN);

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
		//} else {run_400.sh.35466921.out
		//    printf("Error opening timings output file\n");
		//}
		}


std::string* read_data(const char* filename, int& numGenes, int& numSamples, int& numTumor, int& numNormal,
				std::set<int>& tumorSamples, std::vector<std::set<int>>& sparseTumorData,
				std::vector<std::set<int>>& sparseNormalData, int rank) {

		MPI_Status status;
		char *file_buffer = nullptr;
		MPI_Offset file_size = 0;

		MPI_File dataFile;
		int rc = MPI_File_open(MPI_COMM_WORLD, const_cast<char*>(filename), MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile);
		if (rc != MPI_SUCCESS) {
				char error_string[BUFSIZ];
				int length_of_error_string;
				MPI_Error_string(rc, error_string, &length_of_error_string);
				fprintf(stderr, "Rank %d: Error opening file: %s\n", rank, error_string);
				MPI_Abort(MPI_COMM_WORLD, rc);
		}

		MPI_File_get_size(dataFile, &file_size);

		file_buffer = new char[file_size + 1];
		file_buffer[file_size] = '\0';

		MPI_File_read_all(dataFile, file_buffer, file_size, MPI_CHAR, &status);

		MPI_File_close(&dataFile);

		char *line = strtok(file_buffer, "\n");
		if (line == NULL) {
				fprintf(stderr, "Rank %d: No lines in file\n", rank);
				MPI_Abort(MPI_COMM_WORLD, 1);
		}

		int value;
		if (sscanf(line, "%d %d %d %d %d", &numGenes, &numSamples, &value, &numTumor, &numNormal) != 5) {
				fprintf(stderr, "Rank %d: Error reading the first line numbers\n", rank);
				MPI_Abort(MPI_COMM_WORLD, 1);
		}

		sparseTumorData.resize(numGenes);
		sparseNormalData.resize(numGenes);
		std::string* geneIdArray = new std::string[numGenes];

		line = strtok(NULL, "\n");
		while (line != NULL) {
				int gene, sample, val;
				char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];

				if (sscanf(line, "%d %d %d %s %s", &gene, &sample, &val, geneId, sampleId) != 5) {
						fprintf(stderr, "Rank %d: Error reading data line: %s\n", rank, line);
						delete[] file_buffer;
						MPI_Abort(MPI_COMM_WORLD, 1);
				}

				geneIdArray[gene] = geneId;

				if (val > 0) {
						if (sample < numTumor) {
								sparseTumorData[gene].insert(sample);
								tumorSamples.insert(sample);
						} else {
								sparseNormalData[gene].insert(sample);
						}
				}

				line = strtok(NULL, "\n");
		}

		delete[] file_buffer;
		return geneIdArray;

}

void master_process(int num_workers, long long int num_Comb) {
		long long int next_idx = num_workers * CHUNK_SIZE;
		while (next_idx < num_Comb) {
				auto now = std::chrono::system_clock::now();
				time_t now_time = std::chrono::system_clock::to_time_t(now);
				printf("Master currently holding next_idx: %lld out of num_Comb: %lld at time: %s\n", next_idx, num_Comb, std::ctime(&now_time));
				fflush(stdout);
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

double worker_process(int rank, long long int num_Comb,
				std::vector<std::set<int>>& tumorData,
				const std::vector<std::set<int>>& normalData,
				int numGenes, long long int& count, int Nt, int Nn, const char* hit3_file, double& localBestMaxF, std::array<int, 4>& localComb) {

		long long int begin = (rank - 1) * CHUNK_SIZE;
		long long int end = std::min(begin + CHUNK_SIZE, num_Comb);
		MPI_Status status;
		double process_lambda_time = 0.0;
		while (end <= num_Comb) {
				double process_start_time = MPI_Wtime();
				process_lambda_interval(tumorData, normalData, begin, end, numGenes, count, localComb, Nt, Nn, localBestMaxF);
				double process_end_time = MPI_Wtime();
				process_lambda_time += (process_end_time - process_start_time);

				char c = 'a';
				MPI_Send(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

				long long int next_idx;
				MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);
				if (next_idx == -1) break;

				begin = next_idx;
				end = std::min(next_idx + CHUNK_SIZE, num_Comb);
		}
		return process_lambda_time;
}



void distribute_tasks(int rank, int size, int numGenes,
				std::vector<std::set<int>>& tumorData,
				std::vector<std::set<int>>& normalData, long long int& count,
				int Nt, int Nn, const char* outFilename, const char* hit3_file, const std::set<int>& tumorSamples, std::string* geneIdArray) {
		double start_time, end_time;
		double time_calculate_combinations = 0.0;
		double time_master_process = 0.0;
		double time_worker_process = 0.0;
		double time_allreduce = 0.0;
		double time_broadcast = 0.0;
		double time_intersections = 0.0;
		double time_output_write = 0.0;
		double local_process_lambda_time = 0.0;

		start_time = MPI_Wtime();
		long long int num_Comb = nCr(numGenes, 3);
		end_time = MPI_Wtime();
		time_calculate_combinations += end_time - start_time;

		std::set<int> droppedSamples;
		while (tumorSamples != droppedSamples) {
				std::array<int, 4> localComb = {-1, -1, -1, -1};
				double localBestMaxF = -1.0;

				if (rank == 0) { // Master
						start_time = MPI_Wtime();
						master_process(size - 1, num_Comb);
						end_time = MPI_Wtime();
						time_master_process += end_time - start_time;
				} else { // Worker
						start_time = MPI_Wtime();
						local_process_lambda_time += worker_process(rank, num_Comb, tumorData, normalData,
										numGenes, count, Nt, Nn, hit3_file, localBestMaxF, localComb);
						end_time = MPI_Wtime();
						time_worker_process += end_time - start_time;
				}

				struct {
						double value;
						int rank;
				} localResult, globalResult;

				localResult.value = localBestMaxF;
				localResult.rank = rank;

				start_time = MPI_Wtime();
				MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
				end_time = MPI_Wtime();
				time_allreduce += end_time - start_time;

				std::array<int, 4> globalBestComb;
				if (rank == globalResult.rank) {
						globalBestComb = localComb;
				}

				start_time = MPI_Wtime();
				MPI_Bcast(globalBestComb.data(), 4, MPI_INT, globalResult.rank, MPI_COMM_WORLD);
				end_time = MPI_Wtime();
				time_broadcast += end_time - start_time;

				start_time = MPI_Wtime();
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
				end_time = MPI_Wtime();
				time_intersections += end_time - start_time;

				if (rank == 0) {
						start_time = MPI_Wtime();
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
						end_time = MPI_Wtime();
						time_output_write += end_time - start_time;
				}
		}

		// Gather timing data
		double local_times[8] = {
				time_calculate_combinations,
				time_master_process,
				time_worker_process,
				local_process_lambda_time,
				time_allreduce,
				time_broadcast,
				time_intersections,
				time_output_write
		};

		double all_times[size][8];
		MPI_Gather(local_times, 8, MPI_DOUBLE, all_times, 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (rank == 0) {
				std::cout << "Granular Timing Information for distribute_tasks:\n";
				int longest_worker = -1;
				int shortest_worker = -1;
				for (int i = 0; i < 8; ++i) {
						double max_time = -1.0, min_time = 1e9, total_time = 0.0;
						int rank_max = 0, rank_min = 0;
						for (int j = 0; j < size; ++j) {
								double time = all_times[j][i];
								if (i == 3)
								{
										max_time = all_times[longest_worker][i];	
										min_time = all_times[shortest_worker][i];	
								}
								else{
										if (time > max_time) {
												max_time = time;
												rank_max = j;
												if (i == 2) longest_worker = j;
										}
										if (time < min_time) {
												min_time = time;
												rank_min = j;
												if (i == 2) shortest_worker = j;
										}
								}
								total_time += time;
						}
						double avg_time = total_time / size;

						switch (i) {
								case 0: std::cout << "Calculating combinations:\n"; break;
								case 1: std::cout << "Master process:\n"; break;
								case 2: std::cout << "Process Summary process from longest worker:\n"; break;
								case 3: std::cout << "Worker process:\n"; break;
								case 4: std::cout << "Allreduce:\n"; break;
								case 5: std::cout << "Broadcast:\n"; break;
								case 6: std::cout << "Intersections:\n"; break;
								case 7: std::cout << "Output writing:\n"; break;
						}
						std::cout << "    Rank " << rank_max << " took the longest time in Process Worker: " << max_time << " seconds.\n";
						std::cout << "    Rank " << rank_min << " took the shortest time in Process Worker: " << min_time << " seconds.\n";
						std::cout << "    Average time in Process Worker: " << avg_time << " seconds.\n";
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
		if (rank == 0){
				printf("Normal data size: %zu, Tumor Data side: %zu\n", normalData.size(), tumorData.size());
				fflush(stdout);
		}
		std::string* geneIdArray = read_data(argv[1], numGenes, numSamples, numTumor, numNormal, tumorSamples, tumorData, normalData, rank);
		if (rank == 0){
				printf("Normal data size: %zu, Tumor Data side: %zu\n", normalData.size(), tumorData.size());
				fflush(stdout);
		}
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

