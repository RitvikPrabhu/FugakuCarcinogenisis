#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "data.h"

#define ceil_div(a, b) (((a) + (b) - 1) / (b))

void db_alloc(struct db_t *db, int num_rows, int num_cols) {
  db->num_rows = num_rows;
  db->num_cols = num_cols;
  const int elem_bits = sizeof(db->data[0]) * CHAR_BIT;
  db->row_elems = ceil_div(num_cols, elem_bits);
  /* printf("num_cols: %d, row_size: %d\n", num_cols, db.row_size); */
  db->data = calloc(db->num_rows * db->row_elems, sizeof(db->data[0]));
}

void db_set(struct db_t *db, int row, int col) {
  const int elem_bits = sizeof(db->data[0]) * CHAR_BIT;

  int bit_idx = (col % elem_bits);
  data_t mask = ((data_t)1) << bit_idx;

  int elem_idx = (col / elem_bits);
  size_t idx = row * db->row_elems + elem_idx;
  /* printf("row: %5d, col: %5d, mask: %016lx, idx: %5lu col/elem_size: %d\n",
   * row, */
  /*        col, mask, idx, col / elem_bits); */
  db->data[idx] |= mask;
}

void get_db(const char *filename, struct db_t *tumor, struct db_t *normal,
            struct db_t *target) {
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
  db_alloc(tumor, num_genes, num_tumor);
  db_alloc(normal, num_genes, num_normal);
  db_alloc(target, 1, num_tumor);
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
