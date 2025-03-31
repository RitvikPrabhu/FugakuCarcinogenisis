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

void write_combination_count(const double *all_values, int count,
                             sets_t dataTable) {

  double sum = 0.0;
  for (int i = 0; i < count; ++i) {
    sum += all_values[i];
  }

  LAMBDA_TYPE possible_combinations = nCr(dataTable.numRows, NUMHITS);

  std::cout << "===== Number of Combinations visited =====\n";
  std::cout << "Pruned number of combinations: " << sum << " combinations\n";
  std::cout << "Total possible number of combinations: "
            << possible_combinations << " combinations\n\n";
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
  distribute_tasks(rank, size, argv[3], argv[2], elapsed_times, dataTable);
  END_TIMING(total_time, elapsed_times[TOTAL_TIME]);

#ifdef ENABLE_PROFILE
  std::vector<double> all_elapsed_times;
  if (rank == 0) {
    all_elapsed_times.resize(size * TIMING_COUNT);
  }

  MPI_Gather(elapsed_times, TIMING_COUNT, MPI_DOUBLE, all_elapsed_times.data(),
             TIMING_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {

    std::vector<double> comboCount;

    for (int r = 0; r < size; r++) {
      comboCount.push_back(
          all_elapsed_times[r * TIMING_COUNT + COMBINATION_COUNT]);
    }

    {
      std::string metricsFilename(argv[2]);
      std::string csvFilename = metricsFilename + ".totalTime";

      std::ofstream csvFile(csvFilename);
      if (!csvFile) {
        std::cerr << "Could not open CSV file: " << csvFilename << std::endl;
      }

      csvFile << "Rank,TOTAL_TIME\n";

      for (int r = 0; r < size; r++) {
        csvFile << r << "," << all_elapsed_times[r * TIMING_COUNT + TOTAL_TIME]
                << "\n";
      }

      csvFile.close();
    }

    write_combination_count(comboCount.data(), comboCount.size(), dataTable);
  }
#endif

  MPI_Finalize();
  return 0;
}
