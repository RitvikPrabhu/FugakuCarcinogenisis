#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include <mpi.h>

#include "data.h"

const char *PRUNED_3HIT_TRIPLETS = "/home/vatai/3hit_pruned_ACC_data_400.bin";
const char *DB_FILE = "/home/vatai/ACC.combinedData.txt";
const size_t CHUNK_SIZE = 1 << 15;

size_t serial_num_triplets(const char *filename) {
  FILE *file;
  size_t num_triplets;
  file = fopen(filename, "r");
  fread(&num_triplets, sizeof(num_triplets), 1, file);
  fclose(file);
  return num_triplets;
}

size_t get_num_triplets(const char *filename) {
  MPI_File file;
  MPI_Status status;
  size_t num_triplets;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                &file);
  MPI_File_read_all(file, &num_triplets, 1, MPI_LONG, &status);
  MPI_File_close(&file);
  return num_triplets;
}

void master() {
  printf("Hello\n");
  size_t num_triplets = serial_num_triplets(PRUNED_3HIT_TRIPLETS);
  printf("num_triplets: %lu\n", num_triplets);
}

void worker() {
  struct db_t tumor, normal;
  // if (rank == 0)
  get_db(DB_FILE, &tumor, &normal);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    master();
  else
    worker();

  MPI_Finalize();
  return 0;
}
