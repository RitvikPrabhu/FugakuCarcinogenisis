#ifndef MPI_SPECIFIC_H
#define MPI_SPECIFIC_H

#include <mpi.h>

#include "constants.h"

void master_process(int num_workers, long long int num_Comb);

MPIResult perform_MPI_allreduce(const MPIResult &localResult);

#endif
