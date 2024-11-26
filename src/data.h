#pragma once
#include <stdint.h>

typedef uint64_t data_t;

struct db_t {
  int num_rows;
  int num_cols;
  int row_elems;
  data_t *data;
};

void db_alloc(struct db_t *db, int num_rows, int num_cols);

void db_set(struct db_t *db, int row, int col);

void get_db(const char *filename, struct db_t *tumor, struct db_t *normal);
