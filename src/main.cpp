#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <unistd.h>
#include <vector>

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

void write_worker_time_metrics(const char *metricsFile, const double *all_times,
                               int count, const char *metric_name) {
  if (count <= 0) {
    std::cerr << "No data points provided for " << metric_name << "\n";
    return;
  }

  double max_val = all_times[0];
  double min_val = all_times[0];
  double sum = 0.0;

  for (int i = 0; i < count; ++i) {
    max_val = std::max(max_val, all_times[i]);
    min_val = std::min(min_val, all_times[i]);
    sum += all_times[i];
  }
  double mean = sum / count;

  double variance = 0.0;
  for (int i = 0; i < count; ++i) {
    double diff = all_times[i] - mean;
    variance += diff * diff;
  }
  variance /= count;
  double stddev = std::sqrt(variance);
  double range = max_val - min_val;

  std::vector<double> sorted_times(all_times, all_times + count);
  std::sort(sorted_times.begin(), sorted_times.end());

  double median;
  if (count % 2 == 0) {
    median = (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2.0;
  } else {
    median = sorted_times[count / 2];
  }

  std::ofstream ofs(metricsFile, std::ios_base::app);
  if (!ofs.is_open()) {
    std::cerr << "Error opening metrics file: " << metricsFile << std::endl;
    return;
  }

  ofs << "===== Metrics for: " << metric_name << " =====\n";
  ofs << "CHUNK SIZE OF: " << CHUNK_SIZE << "\n";
  ofs << "Max: " << max_val << "\n";
  ofs << "Min: " << min_val << "\n";
  ofs << "Median: " << median << "\n";
  ofs << "Range: " << range << "\n";
  ofs << "Mean: " << mean << "\n";
  ofs << "Std Dev: " << stddev << "\n\n";

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

  START_TIMING(total_time);
  distribute_tasks(rank, size, argv[3], elapsed_times, dataTable);
  END_TIMING(total_time, elapsed_times[TOTAL_TIME]);

  std::vector<double> all_elapsed_times;
  if (rank == 0) {
    all_elapsed_times.resize(size * TIMING_COUNT);
  }

  MPI_Gather(elapsed_times, TIMING_COUNT, MPI_DOUBLE, all_elapsed_times.data(),
             TIMING_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<double> worker_times, worker_runTimes, idle_times, total_times;

    for (int r = 1; r < size; r++) {
      worker_times.push_back(all_elapsed_times[r * TIMING_COUNT + WORKER_TIME]);
      worker_runTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + RUNNING_TIME]);
      idle_times.push_back(all_elapsed_times[r * TIMING_COUNT + IDLE_TIME]);
      total_times.push_back(all_elapsed_times[r * TIMING_COUNT + TOTAL_TIME]);
    }

    std::ofstream ofs(argv[2]);
    ofs << "Performance Metrics\n";
    ofs.close();

    write_worker_time_metrics(argv[2], worker_times.data(), worker_times.size(),
                              "WORKER_TIME");
    write_worker_time_metrics(argv[2], worker_runTimes.data(),
                              worker_runTimes.size(), "RUNNING_TIME");
    write_worker_time_metrics(argv[2], idle_times.data(), idle_times.size(),
                              "IDLE_TIME");
    write_worker_time_metrics(argv[2], total_times.data(), total_times.size(),
                              "TOTAL_TIME");
  }

  MPI_Finalize();
  return 0;
}
