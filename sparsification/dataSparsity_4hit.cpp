#include <utility>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>

#define MAX_BUF_SIZE 1024
#define CHUNK_SIZE 5LL


void process_lambda_interval(const std::vector<std::set<int>>& tumorData, long long int startComb, long long int endComb, int totalGenes, long long int &count, int rank, std::vector<std::array<int, 4>>& bestCombinations, const std::vector<std::array<int32_t, 3>>& workload){

    for (const auto& triplet : workload){
		int i = triplet[0];
		int j = triplet[1];
		int k = triplet[2];
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
						//bestCombinations.push_back({i, j, k, l});
						count++;
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

long long int get_triplet_count(const char* filename, int rank) {
    long long int triplet_count = 0;
    if (rank == 0) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Could not open binary file " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.read(reinterpret_cast<char*>(&triplet_count), sizeof(long long int));
        if (infile.fail()) {
            std::cerr << "Error reading triplet count from binary file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile.close();
    }
    MPI_Bcast(&triplet_count, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    return triplet_count;
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

int* worker_process(int rank, int num_workers, long long int num_Comb,
                    const std::vector<std::set<int>>& tumorData,
                    const std::vector<std::set<int>>& normalData,
                    int numGenes, long long int& count, int Nt, int Nn, size_t& data_size, const char* hit3_file) {
    int begin = (rank - 1) * CHUNK_SIZE;
    int end = std::min(begin + CHUNK_SIZE, num_Comb);
    MPI_Status status;
    std::vector<std::array<int, 4>> localComb;

    while (end <= num_Comb) {
		std::vector<std::array<int32_t, 3>> workload = read_triplets_segment(hit3_file, begin, end);
        process_lambda_interval(tumorData, begin, end, numGenes, count, rank, localComb, workload);
		char c = 'a';
        MPI_Send(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

        int next_idx;
        MPI_Recv(&next_idx, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        if (next_idx == -1) break;

        begin = next_idx;
        end = std::min(next_idx + CHUNK_SIZE, num_Comb);
    }

    size_t num_combinations = localComb.size();
    data_size = num_combinations * 4 * sizeof(int);
    int* data_buffer = new int[num_combinations * 4];
    for (size_t idx = 0; idx < num_combinations; ++idx) {
        data_buffer[idx * 4 + 0] = localComb[idx][0];
        data_buffer[idx * 4 + 1] = localComb[idx][1];
        data_buffer[idx * 4 + 2] = localComb[idx][2];
        data_buffer[idx * 4 + 3] = localComb[idx][3];
    }
	
	return data_buffer;
}


void distribute_tasks(int rank, int size, int numGenes,
                      const std::vector<std::set<int>>& tumorData,
                      const std::vector<std::set<int>>& normalData, long long int& count,
                      int Nt, int Nn, const char* filename, const char* hit3_file) {
   

	long long int num_Comb = get_triplet_count(hit3_file, rank); 
    int* data_buffer = nullptr;
    size_t data_size = 0;

    if (rank == 0) { // Master
        master_process(size - 1, num_Comb);
        data_size = 0; // No data buffer to write for rank 0
    } else { // Worker
        data_buffer = worker_process(rank, size - 1, num_Comb, tumorData, normalData,
                                     numGenes, count, Nt, Nn, data_size, hit3_file);
    }

    long long int total_count = 0;
    MPI_Reduce(&count, &total_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	count = total_count;
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_Offset local_offset;
    MPI_Offset total_data_size;

    size_t write_data_size = (rank == 0) ? sizeof(long long int) : data_size;
    MPI_Exscan(&write_data_size, &local_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        local_offset = 0; // For rank 0, start at the beginning
    }

    if (rank == 0) {
        MPI_File_write_at(file, local_offset, &total_count, sizeof(long long int), MPI_BYTE, MPI_STATUS_IGNORE);
    } else {
        local_offset += sizeof(long long int); 
        MPI_File_write_at(file, local_offset, data_buffer, data_size, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&file);

    if (data_buffer != nullptr) {
        delete[] data_buffer;
    }
}
int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5){
        printf("Four argument expected: ./graphSparcity <dataFile> <outputMetricFile> <prunedDataOutputFile> <3hit Pruned File>");
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
	
    long long int totalCount = 0;
    distribute_tasks(rank, size, numGenes, tumorData, normalData, totalCount, numTumor, numNormal, argv[3], argv[4]);
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

