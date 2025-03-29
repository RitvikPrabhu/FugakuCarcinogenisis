#ifndef UTILS_H
#define UTILS_H

#include "commons.h"

#define MAX_BUF_SIZE 1024

enum profile_out {
  WORKER_TIME,
  WORKER_RUNNING_TIME,
  WORKER_IDLE_TIME,

  PROCESS_LAMBDA_GET_ROW,
  PROCESS_LAMBDA_INTERSECT,
  PROCESS_LAMBDA_SET_COUNT,

  MASTER_TIME,

  TOTAL_TIME,
  TIMING_COUNT
};

#ifdef ENABLE_PROFILE
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

#endif
