#include <bitset>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <fstream> 

#include "commons.h"
#include "fourHit.h"
#include "readFile.h"
#include "utils.h"
// ###########################HELPER#########################
bool parse_arguments(int argc, char *argv[]) {
  if (argc < 4) {
    if (argc > 0) { // To ensure argv[0] is valid
      std::cout << "Usage: " << argv[0]
                << " <dataFile> <outputMetricFile> <prunedDataOutputFile>\n";
    } else {
      std::cout << "Usage: <program> <dataFile> <outputMetricFile> "
                   "<prunedDataOutputFile>\n";
    }
    return false;
  }
  return true;
}

void write_worker_time_metrics(const char* metricsFile,
                               double* all_times, int size)
{
    double max_val = all_times[0];
    double min_val = all_times[0];
    double sum     = 0.0;
    for (int i = 0; i < size; i++) {
        if (all_times[i] > max_val) max_val = all_times[i];
        if (all_times[i] < min_val) min_val = all_times[i];
        sum += all_times[i];
    }
    double mean = sum / size;

    double variance = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = all_times[i] - mean;
        variance   += diff * diff;
    }
    variance /= size;     
    double stddev = std::sqrt(variance);
    double range  = max_val - min_val;

    std::ofstream ofs(metricsFile);
    if (!ofs.is_open()) {
        std::cerr << "Error opening metrics file: " << metricsFile << std::endl;
        return;
    }

    ofs << "CHUNK SIZE OF: " << CHUNK_SIZE << "\n"
        << "Max Worker Time: " << max_val << "\n"
        << "Min Worker Time: " << min_val << "\n"
        << "Range: " << range << "\n"
        << "Mean: " << mean << "\n"
        << "Std Dev: " << stddev << "\n";
    ofs.close();
}

// #########################MAIN###########################
int main(int argc, char *argv[]) {

  int rank, size;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!parse_arguments(argc, argv)) {
    MPI_Finalize();
    return 1;
  }
	
  double elapsed_times[TIMING_COUNT] = {0.0};

  sets_t dataTable = read_data(argv[1], rank);

  distribute_tasks(rank, size, argv[3], elapsed_times, dataTable);

  double local_worker_time = elapsed_times[WORKER_TIME];
  std::vector<double> all_times;
  if (rank == 0) {
    all_times.resize(size);
  }

  MPI_Gather(&local_worker_time, 1, MPI_DOUBLE,
             all_times.data(), 1, MPI_DOUBLE,
             0, MPI_COMM_WORLD);

if (rank == 0) {
    write_worker_time_metrics(argv[2], all_times.data(), size);
  }

  MPI_Finalize();
  return 0;
}
