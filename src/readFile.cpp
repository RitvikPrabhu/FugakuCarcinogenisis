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
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Rank %d: Error opening file \"%s\"\n", rank, filename);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  fseek(fp, 0, SEEK_END);
  file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *buffer = new char[file_size + 1];

  size_t bytesRead = fread(buffer, 1, file_size, fp);
  if (bytesRead < (size_t)file_size) {
    fprintf(stderr,
            "Rank %d: Could not read full file (only %zu of %lld bytes)\n",
            rank, bytesRead, (long long)file_size);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  buffer[file_size] = '\0';

  fclose(fp);
  return buffer;
}

void allocate_sets_from_header(sets_t &table, const char *header_line,
                               int rank) {
  long long num_rows, num_cols, unused, numTumor, numNormal;
  if (sscanf(header_line, "%lld %lld %lld %lld %lld", &num_rows, &num_cols,
             &unused, &numTumor, &numNormal) != 5) {
    fprintf(stderr, "Rank %d: Header parsing failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  table.numRows = static_cast<size_t>(num_rows);
  table.numCols = static_cast<size_t>(num_cols);
  table.numTumor = static_cast<size_t>(numTumor);
  table.numNormal = static_cast<size_t>(numNormal);
  table.tumorRowUnits = CEIL_DIV(numTumor, BITS_PER_UNIT);
  table.normalRowUnits = CEIL_DIV(numNormal, BITS_PER_UNIT);

  SET_COLLECTION_NEW(table.tumorData, table.numRows, table.numTumor,
                     table.tumorRowUnits);
  SET_COLLECTION_NEW(table.normalData, table.numRows, table.numNormal,
                     table.normalRowUnits);
}

void parse_and_populate(sets_t &table, char *file_buffer, int rank) {
  char *line = strtok(nullptr, "\n");
  size_t row_index = 0;

  while (line && row_index < table.numRows) {
    if (strlen(line) < table.numCols) {
      fprintf(stderr, "Rank %d: Parsing error at line %zu\n", rank, row_index);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (size_t col = 0; col < table.numCols; ++col) {
      if (line[col] == '1') {
        if (col < table.numTumor) {
          SET_COLLECTION_INSERT(table.tumorData, row_index, col, table.numTumor,
                                table.tumorRowUnits);
        } else {
          size_t normalIdx = col - table.numTumor;
          SET_COLLECTION_INSERT(table.normalData, row_index, normalIdx,
                                table.numNormal, table.normalRowUnits);
        }
      }
    }

    row_index++;
    line = strtok(nullptr, "\n");
  }

  if (row_index != table.numRows) {
    fprintf(stderr, "Rank %d: Expected %zu rows, got %zu\n", rank,
            table.numRows, row_index);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

sets_t read_data(const char *filename, int rank) {
  MPI_Offset file_size;
  char *file_buffer = read_entire_file_into_buffer(filename, file_size, rank);

  char *header_line = strtok(file_buffer, "\n");
  if (!header_line) {
    delete[] file_buffer;
    fprintf(stderr, "Rank %d: Empty or malformed file\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  sets_t table;
  allocate_sets_from_header(table, header_line, rank);
  parse_and_populate(table, file_buffer, rank);

  delete[] file_buffer;
  return table;
}
