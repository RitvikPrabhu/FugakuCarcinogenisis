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

#ifndef PRINT_FREQ
#define PRINT_FREQ 10000
#endif

enum profile_out {
  WORKER_TIME,
  WORKER_RUNNING_TIME,
  WORKER_IDLE_TIME,
  MASTER_TIME,

  COMM_GLOBAL_TIME,
  COMM_LOCAL_TIME,

  EXCLUDE_TIME,

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
inline double elapsed_times[TIMING_COUNT] = {0.0};
#define START_TIMING(var) double var##_start = MPI_Wtime();
#define END_TIMING(var, accumulated_time)                                      \
  do {                                                                         \
    double var##_end = MPI_Wtime();                                            \
    accumulated_time += var##_end - var##_start;                               \
  } while (0)

inline long long bound_level_counts[NUMHITS] = {0};
#define INCREMENT_BOUND_LEVEL(lvl) (++bound_level_counts[(lvl)])

struct ProgressStats {
  std::size_t dist_iters_completed = 0;
  long long cover_count = 0;
  long long total_tumor = 0;
  double dist_start_ts = 0.0;
  double outer_time_sum = 0.0;
  double inner_start_ts = 0.0;
};
inline ProgressStats gprog;

#else
#define START_TIMING(var)
#define END_TIMING(var, accumulated_time)
#define INCREMENT_BOUND_LEVEL(elapsedTimesArr)
#endif

#endif
