#include <cstddef>
#include <cstdint>

#ifndef CONSTANTS_H
#define CONSTANTS_H

// ############MACROS####################
#ifdef ENABLE_TIMING
#define START_TIMING(var) double var##_start = MPI_Wtime();
#define END_TIMING(var, accumulated_time)                                      \
  do {                                                                         \
    double var##_end = MPI_Wtime();                                            \
    accumulated_time += var##_end - var##_start;                               \
  } while (0)
#else
#define START_TIMING(var)
#define END_TIMING(var, accumulated_time)
#endif
// ############MACROS####################

#define MAX_BUF_SIZE 1024
enum time_stages {
  MASTER_WORKER,
  ALL_REDUCE,
  BCAST,
  OVERALL_FILE_LOAD,
  OVERALL_DISTRIBUTE_FUNCTION,
  OVERALL_TOTAL
};

typedef uint64_t unit_t;

struct sets_t {
  size_t num_rows;
  size_t num_tumor;
  size_t num_normal;
  size_t num_cols;
  unit_t *data;
};
#endif
