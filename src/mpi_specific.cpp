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

MPIResult perform_MPI_allreduce(const MPIResult &localResult) {
  MPIResult globalResult;
  MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                MPI_COMM_WORLD);
  return globalResult;
}

void notify_master_chunk_processed(int master_rank = 0, int tag = 1) {
  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, master_rank, tag, MPI_COMM_WORLD);
}

long long int receive_next_chunk_index(MPI_Status &status, int master_rank = 0,
                                       int tag = 2) {
  long long int next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, master_rank, tag, MPI_COMM_WORLD,
           &status);
  return next_idx;
}