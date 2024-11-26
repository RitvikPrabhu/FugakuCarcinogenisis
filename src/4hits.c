#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include <mpi.h>

#include "data.h"

const char *PRUNED_3HIT_TRIPLETS = "/home/vatai/3hit_pruned_ACC_data_400.bin";
const char *DB_FILE = "/home/vatai/ACC.combinedData.txt";

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

struct interval_t {
  int64_t begin;
  int64_t end;
};

struct interval_t calculate_indices(MPI_Comm comm, int64_t total_count) {
  struct interval_t interval;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int64_t per_process = total_count / size;
  int64_t remainder = total_count % size;
  interval.begin = rank * per_process + (rank < remainder ? rank : remainder);
  interval.end = interval.begin + per_process + (rank < remainder ? 1 : 0);
  return interval;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    printf("Hello\n");

  size_t num_triplets = get_num_triplets(PRUNED_3HIT_TRIPLETS);
  /* printf("num_triplets = %lu\n", num_triplets); */

  MPI_Barrier(MPI_COMM_WORLD);
  struct db_t tumor, normal;
  // if (rank == 0)
  get_db(DB_FILE, &tumor, &normal);
  if (rank == 0)
    printf("Done\n");
  MPI_Finalize();
  return 0;
}
