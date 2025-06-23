#ifndef MULTIHIT_H
#define MULTIHIT_H

#include <array>
#include <mpi.h>
#include <string>

#include "commons.h"
#include "utils.h"

struct MPIResultWithComb {
  double f;
  int comb[NUMHITS];
};

struct MPIResult {
  double value;
  int rank;
};

struct LambdaComputed {
  int i, j;
};

using LAMBDA_TYPE = long long;

struct WorkChunk {
  LAMBDA_TYPE start;
  LAMBDA_TYPE end;
};

struct WorkerState {
  int rank;
  LAMBDA_TYPE current_start;
  LAMBDA_TYPE current_end;
  LAMBDA_TYPE current_position;
  bool is_idle;
  std::chrono::steady_clock::time_point last_update;
};

struct GlobalStealRequest {
  int requester_node;
  LAMBDA_TYPE requester_work_remaining;
};

struct GlobalStealResponse {
  bool has_work;
  WorkChunk work_chunk;
  LAMBDA_TYPE sender_remaining_work;
};

struct WorkAdvertisement {
  int node_id;
  LAMBDA_TYPE total_work;
  std::chrono::steady_clock::time_point timestamp;
};

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable,
                      CommsStruct &comms);
#endif
