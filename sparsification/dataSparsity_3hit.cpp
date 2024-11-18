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

void process_lambda_interval(const std::vector<std::set<int>>& data, long long int startComb, long long int endComb, int totalGenes, long long int &count, int rank, int size, std::set<int>& uniqueNumbers){
    for (long long int lambda = startComb; lambda < endComb; lambda++){
        long long int j = static_cast<long long int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
        long long int i = lambda - (j * (j - 1)) / 2;

        const std::set<int>& gene1Tumor = data[i];
        const std::set<int>& gene2Tumor = data[j];

        std::set<int> intersectTumor1;
        std::set_intersection(gene1Tumor.begin(), gene1Tumor.end(), gene2Tumor.begin(), gene2Tumor.end(), std::inserter(intersectTumor1, intersectTumor1.begin()));

        if (!intersectTumor1.empty()) {
            for (long long int k = j + 1; k < totalGenes; k++){
                const std::set<int>& gene3Tumor = data[k];
                std::set<int> intersectTumor2;
                std::set_intersection(gene3Tumor.begin(), gene3Tumor.end(), intersectTumor1.begin(), intersectTumor1.end(), std::inserter(intersectTumor2, intersectTumor2.begin()));

                if (!intersectTumor2.empty()){
                    count++;
                    uniqueNumbers.insert(i);
                    uniqueNumbers.insert(j);
                    uniqueNumbers.insert(k);
		    printf("%lld %lld %lld\n", i, j, k);
                }
            }
        }
    }
}

void write_timings_to_file(const double all_times[][3], int size, long long int totalCount, std::vector<int> allUniqueNumbers) {
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
        std::set<int> globalUniqueNumbers(allUniqueNumbers.begin(), allUniqueNumbers.end());
        int globalUniqueCount = globalUniqueNumbers.size();
	timingFile << "Total unique numbers: " << globalUniqueCount << std::endl;
        timingFile.close();
    } else {
        printf("Error opening timings output file\n");
    }
}

std::vector<std::set<int>> read_data(const char* filename, int& numGenes, int& numSamples, int& numTumor, int& numNormal) {
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

    std::vector<std::set<int>> sparseData(numGenes);

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
                sparseData[gene].insert(sample);
            }
        }
    }

    fclose(dataFile);
    return sparseData;
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

void worker_process(int rank, int num_workers, long long int num_Comb, const std::vector<std::set<int>>& sparseData, int numGenes, long long int& count, std::set<int>& uniqueNumbers) {
    int begin = (rank - 1) * CHUNK_SIZE;
    int end = std::min(begin + CHUNK_SIZE, num_Comb);
    MPI_Status status;
    while (end <= num_Comb) {
        process_lambda_interval(sparseData, begin, end, numGenes, count, rank, num_workers, uniqueNumbers);
        MPI_Request request;
        char c = 'a';
        MPI_Isend(&c, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &request);
        int next_idx;
        MPI_Recv(&next_idx, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        begin = next_idx;
        if (begin == -1) break;
        end = std::min(next_idx + CHUNK_SIZE, num_Comb);
    }
}

void distribute_tasks(int rank, int size, int numGenes, const std::vector<std::set<int>>& sparseData, long long int& count, std::set<int>& uniqueNumbers) {
    int num_workers = size - 1;
    long long int num_Comb, remainder;
    num_Comb = nCr(numGenes, 2);
    remainder = num_Comb % num_workers;

    if (rank == 0) { // Master
        master_process(num_workers, num_Comb);
    } else { // Worker
        worker_process(rank, num_workers, num_Comb, sparseData, numGenes, count, uniqueNumbers);
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
    std::vector<std::set<int>> sparseData = read_data(argv[1], numGenes, numSamples, numTumor, numNormal);
    end_time = MPI_Wtime();
    elapsed_time_loading = end_time - start_time;



    start_time = MPI_Wtime();
    long long int count = 0;
    std::set<int> uniqueNumbers;
    distribute_tasks(rank, size, numGenes, sparseData, count, uniqueNumbers);
    std::vector<int> localUniqueNumbers(uniqueNumbers.begin(), uniqueNumbers.end());
    int localSize = localUniqueNumbers.size();
    std::vector<int> recvSizes(size);

    // Gather sizes of local unique sets at rank 0
    MPI_Gather(&localSize, 1, MPI_INT, recvSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> recvDispls(size);
    int totalSize = 0;
    if (rank == 0) {
        recvDispls[0] = 0;
        for (int i = 1; i < size; ++i) {
            recvDispls[i] = recvDispls[i - 1] + recvSizes[i - 1];
        }
        totalSize = recvDispls[size - 1] + recvSizes[size - 1];
    }

    std::vector<int> allUniqueNumbers(totalSize);

    // Gather all unique numbers at rank 0
    MPI_Gatherv(localUniqueNumbers.data(), localSize, MPI_INT,
                allUniqueNumbers.data(), recvSizes.data(), recvDispls.data(), MPI_INT, 0, MPI_COMM_WORLD);
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
        write_timings_to_file(all_times, size, totalCount, allUniqueNumbers);
    }



    MPI_Finalize();
    return 0;
}

