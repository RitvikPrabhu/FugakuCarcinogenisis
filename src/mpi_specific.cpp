#include "mpi_specific.h"
#include "constants.h"
#include <iostream>
#include <mpi.h>

void master_process(int num_workers, long long int num_Comb) {
  long long int next_idx = num_workers * CHUNK_SIZE;
  while (next_idx < num_Comb) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag == 1) {
      char c;
      int workerRank = status.MPI_SOURCE;
      MPI_Recv(&c, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);
      if (c == 'a') {
        MPI_Send(&next_idx, 1, MPI_LONG_LONG_INT, workerRank, 2,
                 MPI_COMM_WORLD);
        next_idx += CHUNK_SIZE;
      }
    }
  }
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    long long int term_signal = -1;
    MPI_Send(&term_signal, 1, MPI_LONG_LONG_INT, workerRank, 2, MPI_COMM_WORLD);
  }
}
