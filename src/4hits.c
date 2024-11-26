#include <limits.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const char *PRUNED_3HIT_TRIPLETS = "/home/vatai/3hit_pruned_ACC_data_400.bin";
const char *DB_FILE = "/home/vatai/ACC.combinedData.txt";

size_t get_num_triplets(const char *filename) {
  MPI_File file;
  MPI_Status status;
  size_t num_triplets;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                &file);
  MPI_File_read_all(file, &num_triplets, 1, MPI_LONG, &status);
  MPI_File_close(&file);
  return num_triplets;
}

typedef uint64_t data_t;

struct db_t {
  int num_rows;
  int num_cols;
  int row_size;
  data_t *data;
};

#define ceil_div(a, b) (((a) + (b) - 1) / (b))

struct db_t db_alloc(int num_rows, int num_cols) {
  struct db_t db = {.num_rows = num_rows, .num_cols = num_cols};
  const int elem_bits = sizeof(db.data[0]) * CHAR_BIT;
  db.row_size = ceil_div(num_cols, elem_bits);
  /* printf("num_cols: %d, row_size: %d\n", num_cols, db.row_size); */
  db.data = calloc(db.num_rows * db.row_size, sizeof(db.data[0]));
  return db;
}

void db_set(struct db_t db, int row, int col) {
  const int elem_bits = sizeof(db.data[0]) * CHAR_BIT;
  data_t mask = ((data_t)1) << (col % elem_bits);
  size_t idx = row * db.row_size + (col / elem_bits);
  /* printf("row: %5d, col: %5d, mask: %016lx, idx: %5lu col/elem_size: %d\n",
   * row, */
  /*        col, mask, idx, col / elem_bits); */
  db.data[idx] |= mask;
}

struct db_t get_db(const char *filename) {
  FILE *file;
  int num_genes, gene;
  int num_samples, sample;
  int value;
  char gene_id[1024], sample_id[1024];
  int num_tumor, num_normal;
  file = fopen(filename, "r");
  fscanf(file, "%d %d %d %d %d\n", &num_genes, &num_samples, &value, &num_tumor,
         &num_normal);
  /* printf("num_tumor: %d, num_normal: %d\n", num_tumor, num_normal); */
  struct db_t tumor = db_alloc(num_genes, num_tumor);
  struct db_t normal = db_alloc(num_genes, num_normal);
  size_t num_rows = num_genes * num_samples;
  for (int row = 0; row < num_rows; row++) {
    fscanf(file, "%d %d %d %s %s\n", &gene, &sample, &value, gene_id,
           sample_id);
    if (value)
      if (sample < num_tumor)
        db_set(tumor, gene, sample);
      else
        db_set(normal, gene, sample - num_tumor);
  }
  fclose(file);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    printf("Hello\n");

  size_t num_triplets = get_num_triplets(PRUNED_3HIT_TRIPLETS);
  /* printf("num_triplets = %lu\n", num_triplets); */

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    get_db(DB_FILE);
  MPI_Finalize();
  return 0;
}
