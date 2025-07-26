#ifndef UTILS_H
#define UTILS_H
#include "commons.h"
#include <chrono>
#include <cstdint>
#include <mpi.h>
#define MAX_BUF_SIZE 1024
#define MAX_NAME_LEN 256

#ifndef NUMHITS
#pragma message("NUMHITS was not provided, using default of 4")
#define NUMHITS 4
#endif

#ifndef CHUNK_SIZE
#pragma message("CHUNK_SIZE was not provided, using default of 102400")
#define CHUNK_SIZE 102400
#endif

enum profile_out {
  WORKER_TIME,
  WORKER_RUNNING_TIME,
  WORKER_IDLE_TIME,

  MASTER_TIME,
  TOTAL_TIME,
  COMBINATION_COUNT,
  TIMING_COUNT
};

struct CommsStruct {
  MPI_Comm local_comm;
  MPI_Comm global_comm;
  int local_rank;
  int local_size;
  int global_rank;
  int my_node_id;
  int num_nodes;
  bool is_leader;
};

static int hash_hostname(const char *hostname) {
  uint64_t hash = 0;
  while (*hostname) {
    hash = (hash * 31) ^ (*hostname);
    hostname++;
  }
  return static_cast<int>(hash & 0x7FFFFFFF);
}

#define HIERARCHICAL_COMMS 1
// #undef HIERARCHICAL_COMMS

#ifdef ENABLE_PROFILE
#define START_TIMING(var) double var##_start = MPI_Wtime();
#define END_TIMING(var, accumulated_time)                                      \
  do {                                                                         \
    double var##_end = MPI_Wtime();                                            \
    accumulated_time += var##_end - var##_start;                               \
  } while (0)
#define INCREMENT_COMBO_COUNT(elapsedTimesArr)                                 \
  do {                                                                         \
    (elapsedTimesArr)[COMBINATION_COUNT] += 1.0;                               \
  } while (0)
#else
#define START_TIMING(var)
#define END_TIMING(var, accumulated_time)
#define INCREMENT_COMBO_COUNT(elapsedTimesArr)
#endif

#endif
