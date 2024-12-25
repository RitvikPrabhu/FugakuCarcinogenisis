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
initialize_data_structures(int numGenes,
                           std::vector<std::set<int>> &sparseTumorData,
                           std::vector<std::set<int>> &sparseNormalData) {
  sparseTumorData.resize(numGenes);
  sparseNormalData.resize(numGenes);
  return new std::string[numGenes];
}

void parse_data_lines(char *buffer, int numGenes, int numTumor,
                      std::string *geneIdArray, std::set<int> &tumorSamples,
                      std::vector<std::set<int>> &sparseTumorData,
                      std::vector<std::set<int>> &sparseNormalData, int rank) {
  // Move to the first data line
  char *line = strtok(NULL, "\n");
  int count = 0;
  while (line != NULL) {
    count++;
    int gene, sample, val;
    char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];

    if (sscanf(line, "%d %d %d %s %s", &gene, &sample, &val, geneId,
               sampleId) != 5) {
      fprintf(stderr, "Rank %d: Error reading data line: %s\n", rank, line);
      delete[] buffer;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    geneIdArray[gene] = geneId;

    if (val > 0) {
      if (sample < numTumor) {
        sparseTumorData[gene].insert(sample);
        tumorSamples.insert(sample);
      } else {
        sparseNormalData[gene].insert(sample);
      }
    }

    line = strtok(NULL, "\n");
  }
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
                       std::set<int> &tumorSamples,
                       std::vector<std::set<int>> &sparseTumorData,
                       std::vector<std::set<int>> &sparseNormalData, int rank) {

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
      initialize_data_structures(numGenes, sparseTumorData, sparseNormalData);

  // Parse the remaining data lines
  parse_data_lines(file_buffer, numGenes, numTumor, geneIdArray, tumorSamples,
                   sparseTumorData, sparseNormalData, rank);

  // Clean up
  delete[] file_buffer;

  return geneIdArray;
}

/*
std::string *read_data(const char *filename, int &numGenes, int &numSamples,
                       int &numTumor, int &numNormal,
                       std::set<int> &tumorSamples,
                       std::vector<std::set<int>> &sparseTumorData,
                       std::vector<std::set<int>> &sparseNormalData, int rank) {

  MPI_Status status;
  char *file_buffer = nullptr;
  MPI_Offset file_size = 0;

  MPI_File dataFile;
  int rc = MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename),
                         MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile);
  if (rc != MPI_SUCCESS) {
    char error_string[BUFSIZ];
    int length_of_error_string;
    MPI_Error_string(rc, error_string, &length_of_error_string);
    fprintf(stderr, "Rank %d: Error opening file: %s\n", rank, error_string);
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_File_get_size(dataFile, &file_size);

  file_buffer = new char[file_size + 1];
  file_buffer[file_size] = '\0';

  MPI_File_read_all(dataFile, file_buffer, file_size, MPI_CHAR, &status);

  MPI_File_close(&dataFile);

  char *line = strtok(file_buffer, "\n");
  if (line == NULL) {
    fprintf(stderr, "Rank %d: No lines in file\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int value;
  if (sscanf(line, "%d %d %d %d %d", &numGenes, &numSamples, &value, &numTumor,
             &numNormal) != 5) {
    fprintf(stderr, "Rank %d: Error reading the first line numbers\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  sparseTumorData.resize(numGenes);
  sparseNormalData.resize(numGenes);
  std::string *geneIdArray = new std::string[numGenes];

  line = strtok(NULL, "\n");
  while (line != NULL) {
    int gene, sample, val;
    char geneId[MAX_BUF_SIZE], sampleId[MAX_BUF_SIZE];

    if (sscanf(line, "%d %d %d %s %s", &gene, &sample, &val, geneId,
               sampleId) != 5) {
      fprintf(stderr, "Rank %d: Error reading data line: %s\n", rank, line);
      delete[] file_buffer;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    geneIdArray[gene] = geneId;

    if (val > 0) {
      if (sample < numTumor) {
        sparseTumorData[gene].insert(sample);
        tumorSamples.insert(sample);
      } else {
        sparseNormalData[gene].insert(sample);
      }
    }

    line = strtok(NULL, "\n");
  }

  delete[] file_buffer;
  return geneIdArray;
}*/
