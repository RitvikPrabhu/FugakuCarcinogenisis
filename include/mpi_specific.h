#ifndef MPI_SPECIFIC_H
#define MPI_SPECIFIC_H

#include <mpi.h>

#include "constants.h"

void master_process(int num_workers, long long int num_Comb);

MPIResult perform_MPI_allreduce(const MPIResult &localResult);

long long int receive_next_chunk_index(MPI_Status &status, int master_rank,
                                       int tag);

void notify_master_chunk_processed(int master_rank, int tag);

#endif
