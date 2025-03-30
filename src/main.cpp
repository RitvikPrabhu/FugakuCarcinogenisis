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
#include "readFile.h"
#include "utils.h"

#if NUMHITS == 4
#include "fourHit.h"
#elif NUMHITS == 5
#include "fiveHit.h"
#elif NUMHITS == 6
#include "sixHit.h"
#else
#error "NUMHITS value not supported by the code!"
#endif
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

inline LAMBDA_TYPE nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) = C(n, n-r)

  LAMBDA_TYPE result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

void write_combination_count(const char *metricsFile, const double *all_values,
                             int count, sets_t dataTable) {

  double sum = 0.0;

  for (int i = 0; i < count; ++i) {
    sum += all_values[i];
  }

  LAMBDA_TYPE possible_combinations = nCr(dataTable.numRows, NUMHITS);

  std::ofstream ofs(metricsFile, std::ios_base::app);
  if (!ofs.is_open()) {
    std::cerr << "Error opening metrics file: " << metricsFile << std::endl;
    return;
  }

  ofs << "===== Number of Combinations visited" << " =====\n";
  ofs << "Pruned number of combinations: " << sum << " combinations\n";
  ofs << "Total possible number of combinations: " << possible_combinations
      << " combinations \n\n";

  ofs.close();
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
  ofs << "Max: " << max_val << " seconds \n";
  ofs << "Min: " << min_val << " seconds \n";
  ofs << "Median: " << median << " seconds \n";
  ofs << "Range: " << range << " seconds \n";
  ofs << "Mean: " << mean << " seconds\n";
  ofs << "Std Dev: " << stddev << " seconds \n\n";

  ofs.close();
}

void write_master_time_metrics(const char *metricsFile, const double *all_times,
                               int count, const char *metric_name) {
  if (count <= 0) {
    std::cerr << "No data points provided for " << metric_name << "\n";
    return;
  }

  double time = all_times[0];

  std::ofstream ofs(metricsFile, std::ios_base::app);
  if (!ofs.is_open()) {
    std::cerr << "Error opening metrics file: " << metricsFile << std::endl;
    return;
  }

  ofs << "===== Metrics for: " << metric_name << " =====\n";
  ofs << "CHUNK SIZE OF: " << CHUNK_SIZE << "\n";
  ofs << "Time: " << time << " seconds \n\n";

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

#ifdef ENABLE_PROFILE
  std::vector<double> all_elapsed_times;
  if (rank == 0) {
    all_elapsed_times.resize(size * TIMING_COUNT);
  }

  MPI_Gather(elapsed_times, TIMING_COUNT, MPI_DOUBLE, all_elapsed_times.data(),
             TIMING_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<double> worker_times, worker_runTimes, worker_idletimes,
        processLambda_intersect, processLambda_getRow, processLambda_setCount,
        master_time, total_times;

    std::vector<double> distAllreduceTimes, distSetIntersectTimes,
        distSetUnionTimes, distUpdateCollectionTimes, distSetCountTimes,
        comboCount;

    for (int r = 1; r < size; r++) {
      worker_times.push_back(all_elapsed_times[r * TIMING_COUNT + WORKER_TIME]);
      worker_runTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + WORKER_RUNNING_TIME]);
      worker_idletimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + WORKER_IDLE_TIME]);
      processLambda_intersect.push_back(
          all_elapsed_times[r * TIMING_COUNT + PROCESS_LAMBDA_INTERSECT]);
      processLambda_getRow.push_back(
          all_elapsed_times[r * TIMING_COUNT + PROCESS_LAMBDA_GET_ROW]);
      processLambda_setCount.push_back(
          all_elapsed_times[r * TIMING_COUNT + PROCESS_LAMBDA_SET_COUNT]);
    }

    for (int r = 0; r < size; r++) {
      distAllreduceTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + DIST_ALLREDUCE_TIME]);
      distSetIntersectTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + DIST_SET_INTERSECT_TIME]);
      distSetUnionTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + DIST_SET_UNION_TIME]);
      distUpdateCollectionTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + DIST_UPDATE_COLLECTION_TIME]);
      distSetCountTimes.push_back(
          all_elapsed_times[r * TIMING_COUNT + DIST_SET_COUNT_TIME]);
      comboCount.push_back(
          all_elapsed_times[r * TIMING_COUNT + COMBINATION_COUNT]);
      total_times.push_back(all_elapsed_times[r * TIMING_COUNT + TOTAL_TIME]);
    }
    master_time.push_back(elapsed_times[MASTER_TIME]);

    {
      std::string metricsFilename(argv[2]);
      std::string csvFilename = metricsFilename + ".csv";

      std::ofstream csvFile(csvFilename);
      if (!csvFile) {
        std::cerr << "Could not open CSV file: " << csvFilename << std::endl;
      }

      csvFile << "Rank,WORKER_TIME,WORKER_RUNNING_TIME,WORKER_IDLE_TIME,TOTAL_"
                 "TIME\n";

      for (int r = 0; r < size; r++) {
        csvFile << r << "," << all_elapsed_times[r * TIMING_COUNT + WORKER_TIME]
                << ","
                << all_elapsed_times[r * TIMING_COUNT + WORKER_RUNNING_TIME]
                << "," << all_elapsed_times[r * TIMING_COUNT + WORKER_IDLE_TIME]
                << "," << all_elapsed_times[r * TIMING_COUNT + TOTAL_TIME]
                << "\n";
      }

      csvFile.close();
    }

    std::ofstream ofs(argv[2]);
    ofs << "Performance Metrics\n";
    ofs.close();

    write_worker_time_metrics(argv[2], worker_times.data(), worker_times.size(),
                              "WORKER_TIME");
    write_worker_time_metrics(argv[2], worker_runTimes.data(),
                              worker_runTimes.size(), "WORKER_RUNNING_TIME");
    write_worker_time_metrics(argv[2], worker_idletimes.data(),
                              worker_idletimes.size(), "WORKER_IDLE_TIME");
    write_worker_time_metrics(argv[2], processLambda_intersect.data(),
                              processLambda_intersect.size(),
                              "WORKER_PROCESS_LAMBDA_INTERSECT");
    write_worker_time_metrics(argv[2], processLambda_getRow.data(),
                              processLambda_getRow.size(),
                              "WORKER_PROCESS_LAMBDA_ROW_FETCH");
    write_worker_time_metrics(argv[2], processLambda_setCount.data(),
                              processLambda_setCount.size(),
                              "WORKER_PROCESS_LAMBDA_SET_COUNT");
    write_worker_time_metrics(argv[2], distAllreduceTimes.data(),
                              distAllreduceTimes.size(), "DIST_ALLREDUCE_TIME");
    write_worker_time_metrics(argv[2], distSetIntersectTimes.data(),
                              distSetIntersectTimes.size(),
                              "DIST_SET_INTERSECT_TIME");
    write_worker_time_metrics(argv[2], distSetUnionTimes.data(),
                              distSetUnionTimes.size(), "DIST_SET_UNION_TIME");
    write_worker_time_metrics(argv[2], distUpdateCollectionTimes.data(),
                              distUpdateCollectionTimes.size(),
                              "DIST_UPDATE_COLLECTION_TIME");
    write_worker_time_metrics(argv[2], distSetCountTimes.data(),
                              distSetCountTimes.size(), "DIST_SET_COUNT_TIME");
    write_master_time_metrics(argv[2], master_time.data(), master_time.size(),
                              "MASTER_TIME");
    write_combination_count(argv[2], comboCount.data(), comboCount.size(),
                            dataTable);

    write_worker_time_metrics(argv[2], total_times.data(), total_times.size(),
                              "TOTAL_TIME");
  }
#endif

  MPI_Finalize();
  return 0;
}
