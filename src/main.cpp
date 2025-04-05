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

#if NUMHITS == 2
#include "twoHit.h"
#elif NUMHITS == 3
#include "threeHit.h"
#elif NUMHITS == 4
#include "fourHit.h"
#elif NUMHITS == 5
#include "fiveHit.h"
#elif NUMHITS == 6
#include "sixHit.h"
#elif NUMHITS == 7
#include "sevenHit.h"
#elif NUMHITS == 8
#include "eightHit.h"
#elif NUMHITS == 9
#include "nineHit.h"
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

  double program_time = 0.0;
  double omit_time = 0.0;

  sets_t dataTable = read_data(argv[1], rank);

  START_TIMING(total_time);
  distribute_tasks(rank, size, argv[3], argv[2], dataTable, &omit_time);
  END_TIMING(total_time, program_time);
  program_time -= omit_time;
#ifdef ENABLE_PROFILE
  std::vector<double> all_program_times(size, 0.0);
  MPI_Gather(&program_time, 1, MPI_DOUBLE, all_program_times.data(), 1,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {

    {
      std::string metricsFilename(argv[2]);
      std::string csvFilename = metricsFilename + ".totalTime";

      std::ofstream csvFile(csvFilename);
      if (!csvFile) {
        std::cerr << "Could not open CSV file: " << csvFilename << std::endl;
      }

      csvFile << "Rank,TOTAL_TIME\n";

      for (int r = 0; r < size; r++) {
        csvFile << r << "," << all_program_times[r] << "\n";
      }

      csvFile.close();
    }
  }
#endif

  MPI_Finalize();
  return 0;
}
