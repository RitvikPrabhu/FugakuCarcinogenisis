#include "readFile.h"
#include "commons.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <unistd.h> // for gethostname

#define MAX_NAME_LEN 256

int hash_hostname(const char *hostname) {
  int hash = 0;
  while (*hostname) {
    hash = (hash * 31) ^ (*hostname); // Prime-based hashing with XOR
    hostname++;
  }
  return hash & 0x7FFFFFFF; // Ensure positive value
}

char *broadcast_file_buffer(const char *filename, int rank,
                            size_t &out_file_size, MPI_Comm comm) {
  char *buffer = nullptr;

  MPI_Comm node_comm, leaders_comm;
  int node_rank, node_size;

  char node_name[MAX_NAME_LEN];
  gethostname(node_name, sizeof(node_name));

  int node_color = hash_hostname(node_name);

  MPI_Comm_split(comm, node_color, rank, &node_comm);
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);

  int is_leader = (node_rank == 0) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(comm, is_leader, rank, &leaders_comm);

  if (rank == 0) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
      fprintf(stderr, "Rank 0: Failed to open file %s\n", filename);
      MPI_Abort(comm, 1);
    }
    fseek(fp, 0, SEEK_END);
    out_file_size = ftell(fp);
    rewind(fp);
    buffer = new char[out_file_size + 1];
    size_t bytes_read = fread(buffer, 1, out_file_size, fp);
    fclose(fp);
    if (bytes_read != out_file_size) {
      fprintf(stderr, "Rank 0: File read error\n");
      MPI_Abort(comm, 1);
    }
    buffer[out_file_size] = '\0';
  }

  if (is_leader == 0) { // Only node leaders participate
    MPI_Bcast(&out_file_size, 1, MPI_UNSIGNED_LONG_LONG, 0, leaders_comm);
    if (node_rank == 0 && rank != 0) {
      buffer = new char[out_file_size + 1];
    }
    MPI_Bcast(buffer, out_file_size, MPI_CHAR, 0, leaders_comm);
    buffer[out_file_size] = '\0';
  }

  MPI_Bcast(&out_file_size, 1, MPI_UNSIGNED_LONG_LONG, 0, node_comm);
  if (node_rank != 0) {
    buffer = new char[out_file_size + 1];
  }
  MPI_Bcast(buffer, out_file_size, MPI_CHAR, 0, node_comm);
  buffer[out_file_size] = '\0';

  MPI_Comm_free(&node_comm);
  if (is_leader == 0) { // Only leaders have valid leaders_comm
    MPI_Comm_free(&leaders_comm);
  }

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
        if (col < table.numTumor)
          SET_COLLECTION_INSERT(table.tumorData, row_index, col, table.numTumor,
                                table.tumorRowUnits);
        else
          SET_COLLECTION_INSERT(table.normalData, row_index,
                                col - table.numTumor, table.numNormal,
                                table.normalRowUnits);
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
  sets_t table = {0};
  size_t file_size;
  char *file_buffer =
      broadcast_file_buffer(filename, rank, file_size, MPI_COMM_WORLD);
  char *header_line = strtok(file_buffer, "\n");
  if (!header_line) {
    fprintf(stderr, "Rank %d: Malformed file, missing header\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  allocate_sets_from_header(table, header_line, rank);
  parse_and_populate(table, file_buffer, rank);
  delete[] file_buffer;
  return table;
}
