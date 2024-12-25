#include <iostream>
#include <mpi.h>

#include "constants.h"
#include "mpi_specific.h"
// #########################HELPER###########################

long long int calculate_initial_index(int num_workers) {
  return num_workers * CHUNK_SIZE;
}

void receive_worker_message(char &message, int workerRank) {
  MPI_Status status;
  MPI_Recv(&message, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);
}

void send_chunk_assignment(int workerRank, long long int chunk) {
  MPI_Send(&chunk, 1, MPI_LONG_LONG_INT, workerRank, 2, MPI_COMM_WORLD);
}

void send_termination_signals(int num_workers) {
  long long int termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

void distribute_work(int num_workers, long long int num_Comb,
                     long long int &next_idx) {

  while (next_idx < num_Comb) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      int workerRank = status.MPI_SOURCE;
      char message;
      receive_worker_message(message, workerRank);

      if (message == 'a') {
        send_chunk_assignment(workerRank, next_idx);
        next_idx += CHUNK_SIZE;
      }
    }
  }
}

// #########################MAIN###########################
void master_process(int num_workers, long long int num_Comb) {
  long long int next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);
  send_termination_signals(num_workers);
}
/*
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
}*/
