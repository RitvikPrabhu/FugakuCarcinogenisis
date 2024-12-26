#include <bitset>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <vector>

#include "constants.h"
#include "four_hit.h"
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

void initialize_mpi(int &rank, int &size) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
}

void gather_and_write_timings(int rank, int size, double elapsed_times[],
                              const char *outputMetricFile) {
  double all_times[size][6];

  MPI_Gather(elapsed_times, 6, MPI_DOUBLE, all_times, 6, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    write_timings_to_file(all_times, size, outputMetricFile);
  }
}

void cleanup(unsigned long long *tumorSamples, unsigned long long **tumorData,
             unsigned long long **normalData) {
  delete[] tumorData;
  delete[] normalData;
  delete[] tumorSamples;
  MPI_Finalize();
}

std::string to_binary_string(unsigned long long value, int bits = 64) {
  std::bitset<64> bits_set(value);
  return bits_set.to_string().substr(64 - bits);
}

// #########################MAIN###########################
int main(int argc, char *argv[]) {

  int rank, size;
  initialize_mpi(rank, size);
  if (!parse_arguments(argc, argv)) {
    MPI_Finalize();
    return 1;
  }

  const char *dataFile = argv[1];
  const char *outputMetricFile = argv[2];
  const char *prunedDataOutputFile = argv[3];

  START_TIMING(overall_execution)

  double elapsed_time_loading = 0.0;
  double elapsed_time_func = 0.0;
  double elapsed_time_total = 0.0;
  double elapsed_times[6] = {0.0};

  START_TIMING(loading)
  int numGenes, numSamples, numTumor, numNormal;
  unsigned long long *tumorSamples = nullptr;
  unsigned long long **tumorData = nullptr;
  unsigned long long **normalData = nullptr;

  std::string *geneIdArray =
      read_data(argv[1], numGenes, numSamples, numTumor, numNormal,
                tumorSamples, tumorData, normalData, rank);
  END_TIMING(loading, elapsed_time_loading);

  if (rank == 0) {
    std::cout << "Number of Genes: " << numGenes << "\n";
    std::cout << "Number of Samples: " << numSamples << "\n";
    std::cout << "Number of Tumor Samples: " << numTumor << "\n";
    std::cout << "Number of Normal Samples: " << numNormal << "\n";

    std::cout << "Tumor Samples Bitmask: ";
    for (size_t unit = 0; unit < calculate_bit_units(numTumor); ++unit) {
      std::cout << to_binary_string(
          tumorSamples[unit],
          std::min(64, numTumor - static_cast<int>(unit * 64)));
    }
    std::cout << "\n";

    std::cout << "First 5 Tumor Data Rows (Bitmask):\n";
    for (int gene = 5; gene < std::min(10, numGenes); gene++) {
      std::cout << "Gene " << geneIdArray[gene] << ": ";
      for (size_t unit = 0; unit < calculate_bit_units(numTumor); ++unit) {
        std::cout << to_binary_string(
            tumorData[gene][unit],
            std::min(64, numTumor - static_cast<int>(unit * 64)));
      }

      std::cout << "\n";
    }
    std::cout << "First 5 Normal Data Rows (Bitmask):\n";
    for (int gene = 0; gene < std::min(5, numGenes); gene++) {
      std::cout << "Gene " << geneIdArray[gene] << ": ";
      for (size_t unit = 0; unit < calculate_bit_units(numNormal); ++unit) {
        std::cout << to_binary_string(
            normalData[gene][unit],
            std::min(64, numNormal - static_cast<int>(unit * 64)));
      }
      std::cout << "\n";
    }
  }

  START_TIMING(function_execution)

  distribute_tasks(rank, size, numGenes, tumorData, normalData, numTumor,
                   numNormal, prunedDataOutputFile, tumorSamples, geneIdArray,
                   elapsed_times);
  END_TIMING(function_execution, elapsed_time_func);

  END_TIMING(overall_execution, elapsed_time_total);

  elapsed_times[OVERALL_FILE_LOAD] = elapsed_time_loading;
  elapsed_times[OVERALL_DISTRIBUTE_FUNCTION] = elapsed_time_func;
  elapsed_times[OVERALL_TOTAL] = elapsed_time_total;

#ifdef ENABLE_TIMING
  gather_and_write_timings(rank, size, elapsed_times, outputMetricFile);
#endif

  cleanup(tumorSamples, tumorData, normalData);

  return 0;
}
