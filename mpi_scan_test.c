#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int input, output;
  input = rank + 1;
  MPI_Exscan(&input, &output, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  printf("rank: %d; result: %d\n", rank, output);
  MPI_Finalize();
  return 0;
}
