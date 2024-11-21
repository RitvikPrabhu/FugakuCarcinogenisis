#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

struct scored_triplet_t {
  double fscore;
  int iter[3];
};

void print_triplet(struct scored_triplet_t t) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("(%d) %f: %d,%d,%d\n", rank, t.fscore, t.iter[0], t.iter[1],
         t.iter[2]);
}

void triplet_max_fun(void *a, void *b, int *len, MPI_Datatype *type) {
  struct scored_triplet_t *aa = a;
  struct scored_triplet_t *bb = b;
  assert(*len == 1); // this should be 1
  for (int i = 0; i < *len; i++) {
    bb[i] = bb[i].fscore > aa[i].fscore ? bb[i] : aa[i];
  }
};

int main(int argc, char *argv[]) {
  int err;
  MPI_Init(&argc, &argv);

  MPI_Op max_op;
  err = MPI_Op_create(triplet_max_fun, 1, &max_op);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  struct scored_triplet_t triplet = {
      5 - rank + 1. / 2,
      10 * rank + 1,
      10 * rank + 2,
      10 * rank + 3,
  };

  MPI_Datatype MPI_SCORED_TRIPLET;
  int blocklen[] = {1, 3};
  MPI_Aint disp[] = {(void *)&triplet.fscore - (void *)&triplet,
                     (void *)&triplet.iter - (void *)&triplet};
  MPI_Datatype type[] = {MPI_DOUBLE, MPI_INT};
  MPI_Type_create_struct(2, blocklen, disp, type, &MPI_SCORED_TRIPLET);
  MPI_Type_commit(&MPI_SCORED_TRIPLET);

  struct scored_triplet_t x;
  print_triplet(triplet);
  MPI_Reduce(&triplet, &x, 1, MPI_SCORED_TRIPLET, max_op, 0, MPI_COMM_WORLD);
  print_triplet(triplet);
  if (rank == 0) {
    printf("result: ");
    print_triplet(x);
  }
  MPI_Finalize();
  return 0;
}
