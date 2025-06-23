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

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable,
                      CommsStruct &comms);
#endif
