#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argc);
  printf("Hello\n");
  MPI_Finalize();
  return 0;
}
