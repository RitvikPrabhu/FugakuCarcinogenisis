#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int input, output;
  if (rank) {
    // rank>0 will use 10*rank as the input.
    // Input should be the size of data which rank will write.
    input = 3 * rank;
  } else {
    // rank == 0 will use 8 (i.e. sizeof(unsigned long long))
    input = 8;
  }
  MPI_Exscan(&input, &output, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // Output should be the offset where it should be writing.
  if (rank == 0) {
    // The result for rank == 0 must be explicitly set!
    output = 0;
  }
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "./output.txt", MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);
  char buf[100];
  char c = rank ? 'a' + rank - 1 : 'x';
  // rank 0 writes 8 'x's
  // rank 1 writes `input` number of 'a's
  // rank 2 writes `input` number of 'b's
  // ..etc..
  for (int i = 0; i < input; i++) {
    buf[i] = c + rank;
  }
  MPI_Status status;
  MPI_File_write_at(file, output, buf, input, MPI_CHAR, &status);
  MPI_File_close(&file);
  printf("rank: %d; result: %d\n", rank, output);
  MPI_Finalize();
  return 0;
}
