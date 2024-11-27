#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include <mpi.h>

#include "data.h"

const char *PRUNED_3HIT_TRIPLETS = "3hit_pruned_ACC_data_400.bin";
const char *DB_FILE = "ACC.combinedData.txt";
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

void master_process(MPI_Comm workers) {
  int workers_size;
  MPI_Comm_size(workers, &workers_size);
  size_t num_triplets = serial_num_triplets(PRUNED_3HIT_TRIPLETS);
  int next_idx = workers_size * CHUNK_SIZE;
  while (next_idx < num_triplets) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag == 1) {
      char c;
      int workerRank = status.MPI_SOURCE;
      MPI_Recv(&c, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);
      MPI_Send(&next_idx, 1, MPI_INT, workerRank, 2, MPI_COMM_WORLD);
      next_idx += CHUNK_SIZE;
    }
  }
  for (int workerRank = 1; workerRank <= workers_size; ++workerRank) {
    int term_signal = -1;
    MPI_Send(&term_signal, 1, MPI_INT, workerRank, 2, MPI_COMM_WORLD);
  }
}

void master(MPI_Comm workers) { printf("Hello\n"); }

void worker(MPI_Comm workers) {
  struct db_t tumor, normal, target;
  // if (rank == 0)
  get_db(DB_FILE, &tumor, &normal, &target);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm workers;
  MPI_Comm_split(MPI_COMM_WORLD, rank == 0, rank, &workers);

  if (rank == 0)
    master(workers);
  else
    worker(workers);

  MPI_Finalize();
  return 0;
}
