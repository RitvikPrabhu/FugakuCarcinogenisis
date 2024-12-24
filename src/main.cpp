#include "constants.h"
#include "four_hit.h"
#include "utils.h"
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <vector>

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 4) {
    printf("Three argument expected: ./graphSparcity <dataFile> "
           "<outputMetricFile> <prunedDataOutputFile>");

    MPI_Finalize();
    return 1;
  }

  double total_start_time = MPI_Wtime();

  double start_time, end_time;
  double elapsed_time_loading, elapsed_time_func, elapsed_time_total;
  double elapsed_times[6] = {0.0};

  start_time = MPI_Wtime();
  int numGenes, numSamples, numTumor, numNormal;
  std::set<int> tumorSamples;
  std::vector<std::set<int>> tumorData;

  std::vector<std::set<int>> normalData;
  if (rank == 0) {
    printf("Normal data size: %zu, Tumor Data side: %zu\n", normalData.size(),
           tumorData.size());
    fflush(stdout);
  }
  std::string *geneIdArray =
      read_data(argv[1], numGenes, numSamples, numTumor, numNormal,
                tumorSamples, tumorData, normalData, rank);
  if (rank == 0) {
    printf("Normal data size: %zu, Tumor Data side: %zu\n", normalData.size(),
           tumorData.size());
    fflush(stdout);
  }
  end_time = MPI_Wtime();
  elapsed_time_loading = end_time - start_time;

  start_time = MPI_Wtime();
  long long int totalCount = 0;
  distribute_tasks(rank, size, numGenes, tumorData, normalData, totalCount,
                   numTumor, numNormal, argv[3], argv[2], tumorSamples,
                   geneIdArray, elapsed_times);
  double total_end_time = MPI_Wtime();
  elapsed_time_total = total_end_time - total_start_time;
  end_time = MPI_Wtime();
  elapsed_time_func = end_time - start_time;

  elapsed_times[OVERALL_FILE_LOAD] = elapsed_time_loading;
  elapsed_times[OVERALL_DISTRIBUTE_FUNCTION] = elapsed_time_func;
  elapsed_times[OVERALL_TOTAL] = elapsed_time_total;
  double all_times[size][6];
  MPI_Gather(elapsed_times, 6, MPI_DOUBLE, all_times, 6, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    write_timings_to_file(all_times, size, totalCount, argv[2]);
  }

  MPI_Finalize();
  return 0;
}
