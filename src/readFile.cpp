#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "commons.h"
#include "readFile.h"

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

char *read_entire_file_into_buffer(const char *filename, MPI_Offset &file_size,
                                   int rank) {
  MPI_File fh;
  int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                         MPI_INFO_NULL, &fh);
  handle_mpi_error(rc, "Error opening file", rank);

  rc = MPI_File_get_size(fh, &file_size);
  handle_mpi_error(rc, "Error getting file size", rank);

  char *buffer = new char[file_size + 1];
  buffer[file_size] = '\0';

  MPI_Status status;
  rc = MPI_File_read_at_all(fh, 0, buffer, file_size, MPI_CHAR, &status);
  handle_mpi_error(rc, "Error reading file", rank);

  rc = MPI_File_close(&fh);
  handle_mpi_error(rc, "Error closing file", rank);

  return buffer;
}

sets_t allocate_sets_from_header(const char *header_line, int rank) {
  long long num_cols = 0, num_rows = 0, not_used = 0, num_tumor = 0,
            num_normal = 0;

  int nvals = sscanf(header_line, "%lld %lld %lld %lld %lld", &num_rows,
                     &num_cols, &not_used, &num_tumor, &num_normal);
  if (nvals != 5) {
    handle_parsing_error("Error parsing header line", rank);
  }

  sets_t table;
  table.numRows = static_cast<size_t>(num_rows);
  table.numTumor = static_cast<size_t>(num_tumor);
  table.numNormal = static_cast<size_t>(num_normal);
  table.numCols = static_cast<size_t>(num_cols);

  INIT_DATA(table);
  return table;
}

void parse_and_populate(sets_t &table, char *file_buffer, int rank) {

  size_t row_index = 0;
  char *line = strtok(nullptr, "\n");

  while (line != nullptr && row_index < table.numRows) {
    if (std::strlen(line) < table.numCols) {
      fprintf(stderr, "Rank %d: Error: line %zu has fewer than %zu bits.\n",
              rank, row_index, table.numCols);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (size_t c = 0; c < table.numCols; c++) {
      if (line[c] == '1') {
        if (c < table.numTumor) {
          SET_TUMOR(table, row_index, c);
        } else {
          SET_NORMAL(table, row_index, c);
        }
      }
    }
    row_index++;
    line = strtok(nullptr, "\n");
  }
  // hello
  if (row_index != table.numRows) {
    fprintf(stderr, "Rank %d: Error: expected %zu data lines but got %zu.\n",
            rank, table.numRows, row_index);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

sets_t read_data(const char *filename, int rank) {
  MPI_Offset file_size;
  char *file_buffer = read_entire_file_into_buffer(filename, file_size, rank);

  char *header_line = strtok(file_buffer, "\n");
  if (!header_line) {
    delete[] file_buffer;
    handle_parsing_error("No lines in file.", rank);
  }
  sets_t table = allocate_sets_from_header(header_line, rank);
  parse_and_populate(table, file_buffer, rank);
  delete[] file_buffer;
  return table;
}
