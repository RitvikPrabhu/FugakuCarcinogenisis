#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "constants.h"
#include "utils.h"

// #########################HELPER###########################
void handle_mpi_error(int rc, const char *context, int rank) {
  if (rc != MPI_SUCCESS) {
    char error_string[BUFSIZ];
    int length_of_error_string;
    MPI_Error_string(rc, error_string, &length_of_error_string);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, context, error_string);
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

void handle_parsing_error(const char *message, int rank, int abort_code = 1) {
  fprintf(stderr, "Rank %d: %s\n", rank, message);
  MPI_Abort(MPI_COMM_WORLD, abort_code);
}

MPI_File open_mpi_file(const char *filename, int rank) {
  MPI_File dataFile;
  int rc = MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename),
                         MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile);
  handle_mpi_error(rc, "Error opening file", rank);
  return dataFile;
}

char *read_file_buffer(MPI_File dataFile, MPI_Offset &file_size, int rank) {
  MPI_File_get_size(dataFile, &file_size);

  // Allocate buffer with an extra byte for the null terminator
  char *file_buffer = new char[file_size + 1];
  file_buffer[file_size] = '\0'; // Null-terminate the buffer

  MPI_Status status;
  int rc =
      MPI_File_read_all(dataFile, file_buffer, file_size, MPI_CHAR, &status);
  handle_mpi_error(rc, "Error reading file", rank);

  return file_buffer;
}

void close_mpi_file(MPI_File &dataFile, int rank) {
  int rc = MPI_File_close(&dataFile);
  handle_mpi_error(rc, "Error closing file", rank);
}

void parse_header(const char *line, int &numGenes, int &numSamples, int &value,
                  int &numTumor, int &numNormal, int rank) {
  if (sscanf(line, "%d %d %d %d %d", &numGenes, &numSamples, &value, &numTumor,
             &numNormal) != 5) {
    handle_parsing_error("Error reading the first line numbers", rank);
  }
}

std::string *
initialize_data_structures(int numGenes, int numNormal, int numTumor,
                           unsigned long long *&tumorSamples,
                           unsigned long long **&sparseTumorData,
                           unsigned long long **&sparseNormalData) {
  size_t tumorUnits = calculate_bit_units(numTumor);
  size_t normalUnits = calculate_bit_units(numNormal);

  tumorSamples = new unsigned long long[tumorUnits];
  memset(tumorSamples, 0, tumorUnits * sizeof(unsigned long long));

  sparseTumorData = new unsigned long long *[numGenes];
  sparseNormalData = new unsigned long long *[numGenes];

  for (int gene = 0; gene < numGenes; ++gene) {
    // Allocate and initialize each gene's bit array
    sparseTumorData[gene] = new unsigned long long[tumorUnits];
    memset(sparseTumorData[gene], 0, tumorUnits * sizeof(unsigned long long));

    sparseNormalData[gene] = new unsigned long long[normalUnits];
    memset(sparseNormalData[gene], 0, normalUnits * sizeof(unsigned long long));
  }

  return new std::string[numGenes];
}

void parse_data_lines(char *buffer, int numGenes, int numTumor,
                      std::string *geneIdArray,
                      unsigned long long *tumorSamples,
                      unsigned long long **sparseTumorData,
                      unsigned long long **sparseNormalData, int rank) {
  // Move to the first data line
  char *line = strtok(NULL, "\n");
  while (line != NULL) {
    int gene, sample, val;
    char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];

    if (sscanf(line, "%d %d %d %s %s", &gene, &sample, &val, geneId,
               sampleId) != 5) {
      fprintf(stderr, "Rank %d: Error reading data line: %s\n", rank, line);
      delete[] buffer;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    geneIdArray[gene] = geneId;

    if (sample < numTumor) {
      size_t unit = sample / 64;
      size_t bit = sample % 64;
      sparseTumorData[gene][unit] |= (1ULL << bit);
      tumorSamples[unit] |= (1ULL << bit);
    } else {
      size_t normalSample = sample - numTumor;
      size_t unit = normalSample / 64;
      size_t bit = normalSample % 64;
      sparseNormalData[gene][unit] |= (1ULL << bit);
    }

    line = strtok(NULL, "\n");
  }
}

size_t calculate_bit_units(size_t numGenes) {
  const size_t BITS_PER_UNIT = 64;
  return (numGenes + BITS_PER_UNIT - 1) / BITS_PER_UNIT;
}

// #########################MAIN###########################

long long int nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) == C(n, n-r)

  long long int result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

void write_timings_to_file(const double all_times[][6], int size,
                           const char *filename) {
  std::ofstream timingFile(filename);
  if (!timingFile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing.\n";
    return;
  }

  // CSV header
  timingFile << "RANK,MASTER_WORKER,ALL_REDUCE,BCAST,OVERALL_FILE_LOAD,OVERALL_"
                "DISTRIBUTE_FUNCTION,OVERALL_TOTAL\n";

  // Write each rank's data
  for (int rank = 0; rank < size; ++rank) {
    timingFile << rank;
    for (int stage = 0; stage < 6; ++stage) {
      timingFile << "," << all_times[rank][stage];
    }
    timingFile << "\n";
  }

  timingFile.close();
}

std::string *read_data(const char *filename, int &numGenes, int &numSamples,
                       int &numTumor, int &numNormal,
                       unsigned long long *&tumorSamples,
                       unsigned long long **&sparseTumorData,
                       unsigned long long **&sparseNormalData, int rank) {
  MPI_File dataFile = open_mpi_file(filename, rank);

  // Read the file buffer
  MPI_Offset file_size;
  char *file_buffer = read_file_buffer(dataFile, file_size, rank);

  // Close the MPI file as it's no longer needed
  close_mpi_file(dataFile, rank);

  // Parse the header line
  char *header_line = strtok(file_buffer, "\n");
  if (header_line == NULL) {
    delete[] file_buffer;
    handle_parsing_error("No lines in file", rank);
  }

  int value;
  parse_header(header_line, numGenes, numSamples, value, numTumor, numNormal,
               rank);

  // Initialize data structures
  std::string *geneIdArray =
      initialize_data_structures(numGenes, numNormal, numTumor, tumorSamples,
                                 sparseTumorData, sparseNormalData);

  // Parse the remaining data lines
  parse_data_lines(file_buffer, numGenes, numTumor, geneIdArray, tumorSamples,
                   sparseTumorData, sparseNormalData, rank);

  // Clean up
  delete[] file_buffer;

  return geneIdArray;
}
