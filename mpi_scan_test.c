#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int input, output;
  if (rank) {
    // rank>0 will use 10*rank as the input.
    input = 10 * rank;
  } else {
    // rank == 0 will use 8 (i.e. sizeof(unsigned long long))
    input = 8;
  }
  MPI_Exscan(&input, &output, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    // The result for rank == 0 must be explicitly set!
    output = 0;
  }
  printf("rank: %d; result: %d\n", rank, output);
  MPI_Finalize();
  return 0;
}
