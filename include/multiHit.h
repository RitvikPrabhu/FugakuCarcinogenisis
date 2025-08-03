#ifndef MULTIHIT_H
#define MULTIHIT_H

#include <array>
#include <mpi.h>
#include <string>

#include "commons.h"
#include "utils.h"

const int UPDATE_CHUNK = 20;
const int NUM_RETRIES = 3;

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

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable,
                      CommsStruct &comms);

static inline void process_lambda_interval(LAMBDA_TYPE startComb,
                                           LAMBDA_TYPE endComb,
                                           int bestCombination[], double &maxF,
                                           sets_t &dataTable, SET *buffers,
                                           double elapsed_times[],
                                           CommsStruct &comms);

enum : int {
  TAG_REQUEST_WORK = 10,
  TAG_ASSIGN_WORK = 11,
  TAG_UPDATE_END = 12,
  TAG_UPDATE_START = 13,
  TAG_NODE_STEAL_REQ = 20,
  TAG_NODE_STEAL_REPLY = 21,
  TAG_TOKEN = 31,
  TAG_TERMINATE = 32
};

enum Colour { WHITE = 0, BLACK = 1 };
struct Token {
  int colour;
  bool finalRound;
};
#endif
