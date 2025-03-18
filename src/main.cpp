#include <bitset>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <unistd.h>
#include <vector>

#include "commons.h"
#include "fourHit.h"
#include "readFile.h"

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

/**
void gather_and_write_timings(int rank, int size, double elapsed_times[],
                              const char *outputMetricFile) {
  double all_times[size][6];

  MPI_Gather(elapsed_times, 6, MPI_DOUBLE, all_times, 6, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    write_timings_to_file(all_times, size, outputMetricFile);
  }
}**/

/*void cleanup(sets_t dataTable) {
  delete[] dataTable.normalData;
  delete[] dataTable.tumorData;
} */

// #########################MAIN###########################
int main(int argc, char *argv[]) {

  int rank, size;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // std::cerr << "Process rank " << rank << " PID: " << getpid() << std::endl;
  // sleep(30);

  if (!parse_arguments(argc, argv)) {
    MPI_Finalize();
    return 1;
  }

  // START_TIMING(overall_execution)

  double elapsed_time_loading = 0.0;
  double elapsed_time_func = 0.0;
  double elapsed_time_total = 0.0;
  double elapsed_times[6] = {0.0};

  // START_TIMING(loading)

  sets_t dataTable = read_data(argv[1], rank);

  // END_TIMING(loading, elapsed_time_loading);

  // START_TIMING(function_execution)
  distribute_tasks(rank, size, argv[3], elapsed_times, dataTable);
  // END_TIMING(function_execution, elapsed_time_func);

  // END_TIMING(overall_execution, elapsed_time_total);

  // elapsed_times[OVERALL_FILE_LOAD] = elapsed_time_loading;
  // elapsed_times[OVERALL_DISTRIBUTE_FUNCTION] = elapsed_time_func;
  // elapsed_times[OVERALL_TOTAL] = elapsed_time_total;

#ifdef ENABLE_TIMING
  // gather_and_write_timings(rank, size, elapsed_times, argv[2]);
#endif

  // cleanup(dataTable);
  MPI_Finalize();
  return 0;
}
