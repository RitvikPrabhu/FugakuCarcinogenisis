#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <vector>

#include "fourHit.h"

inline LAMBDA_TYPE calculate_initial_index(int num_workers) {
  return static_cast<LAMBDA_TYPE>(num_workers) * CHUNK_SIZE;
}

inline void distribute_work(int num_workers, LAMBDA_TYPE num_Comb,
                            LAMBDA_TYPE &next_idx) {
  while (next_idx < num_Comb) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag) {
      int workerRank = status.MPI_SOURCE;
      char message;
      MPI_Recv(&message, 1, MPI_CHAR, workerRank, 1, MPI_COMM_WORLD, &status);

      if (message == 'a') {
        MPI_Send(&next_idx, 1, MPI_LONG_LONG_INT, workerRank, 2,
                 MPI_COMM_WORLD);
        next_idx += CHUNK_SIZE;
      }
    }
  }
}

inline void master_process(int num_workers, LAMBDA_TYPE num_Comb) {
  LAMBDA_TYPE next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);

  LAMBDA_TYPE termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

inline LAMBDA_TYPE nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) = C(n, n-r)

  LAMBDA_TYPE result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

std::pair<LAMBDA_TYPE, LAMBDA_TYPE>
calculate_initial_chunk(int rank, LAMBDA_TYPE num_Comb,
                        LAMBDA_TYPE chunk_size) {
  LAMBDA_TYPE begin = (rank - 1) * chunk_size;
  LAMBDA_TYPE end = std::min(begin + chunk_size, num_Comb);
  return {begin, end};
}

inline LambdaComputed compute_lambda_variables(LAMBDA_TYPE lambda,
                                               int totalGenes) {
  LambdaComputed computed;
  computed.j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
  if (computed.j > totalGenes - (NUMHITS - 2)) {
    computed.j = -1;
    return computed;
  }
  computed.i = lambda - (computed.j * (computed.j - 1)) / 2;
  return computed;
}

void write_output(int rank, std::ofstream &outfile,
                  const std::array<int, NUMHITS> &globalBestComb,
                  double F_max) {
  outfile << "(";
  for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
    outfile << globalBestComb[idx];
    if (idx != globalBestComb.size() - 1) {
      outfile << ", ";
    }
  }
  outfile << ")  F-max = " << F_max << std::endl;
}

void max_f_with_comb(void *in, void *inout, int *len, MPI_Datatype *type) {
  const MPIResultWithComb *in_vals = static_cast<const MPIResultWithComb *>(in);
  MPIResultWithComb *inout_vals = static_cast<MPIResultWithComb *>(inout);

  for (int i = 0; i < *len; i++) {
    if (in_vals[i].f > inout_vals[i].f) {
      inout_vals[i].f = in_vals[i].f;
      for (int j = 0; j < NUMHITS; j++) {
        inout_vals[i].comb[j] = in_vals[i].comb[j];
      }
    }
  }
}

MPI_Op create_max_f_with_comb_op(MPI_Datatype MPI_RESULT_WITH_COMB) {
  MPI_Op MPI_MAX_F_WITH_COMB;
  MPI_Op_create(&max_f_with_comb, 1, &MPI_MAX_F_WITH_COMB);
  return MPI_MAX_F_WITH_COMB;
}

MPI_Datatype create_mpi_result_with_comb_type() {
  MPI_Datatype MPI_RESULT_WITH_COMB;

  const int nitems = 2;
  int blocklengths[] = {1, NUMHITS};
  MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT};

  MPI_Aint offsets[nitems];
  offsets[0] = offsetof(MPIResultWithComb, f);
  offsets[1] = offsetof(MPIResultWithComb, comb);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &MPI_RESULT_WITH_COMB);
  MPI_Type_commit(&MPI_RESULT_WITH_COMB);

  return MPI_RESULT_WITH_COMB;
}

void process_lambda_interval(LAMBDA_TYPE startComb, LAMBDA_TYPE endComb,
                             std::array<int, NUMHITS> &bestCombination,
                             double &maxF, sets_t &dataTable,
                             SET &intersectionBuffer, SET &scratchBufferij,
                             SET &scratchBufferijk) {
  double alpha = 0.1;
  int totalGenes = dataTable.numRows;

  size_t tumorBitsPerRow = dataTable.tumorRowUnits * BITS_PER_UNIT;
  size_t normalBitsPerRow = dataTable.normalRowUnits * BITS_PER_UNIT;

  for (LAMBDA_TYPE lambda = startComb; lambda <= endComb; lambda++) {
    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
    if (computed.j < 0) {
      continue;
    }

    SET rowI =
        GET_ROW(dataTable.tumorData, computed.i, dataTable.tumorRowUnits);
    SET rowJ =
        GET_ROW(dataTable.tumorData, computed.j, dataTable.tumorRowUnits);

    SET_INTERSECT(scratchBufferij, rowI, rowJ, tumorBitsPerRow);

    if (SET_IS_EMPTY(scratchBufferij, tumorBitsPerRow)) {
      continue;
    }

    for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {
      SET rowK = GET_ROW(dataTable.tumorData, k, dataTable.tumorRowUnits);

      SET_INTERSECT(scratchBufferijk, scratchBufferij, rowK, tumorBitsPerRow);

      if (SET_IS_EMPTY(scratchBufferijk, tumorBitsPerRow)) {
        continue;
      }

      for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
        SET rowL = GET_ROW(dataTable.tumorData, l, dataTable.tumorRowUnits);

        SET_INTERSECT(intersectionBuffer, scratchBufferijk, rowL,
                      tumorBitsPerRow);
        if (SET_IS_EMPTY(intersectionBuffer, tumorBitsPerRow)) {
          continue;
        }

        int TP = SET_COUNT(intersectionBuffer, tumorBitsPerRow);

        SET rowIN =
            GET_ROW(dataTable.normalData, computed.i, dataTable.normalRowUnits);
        SET rowJN =
            GET_ROW(dataTable.normalData, computed.j, dataTable.normalRowUnits);
        SET rowKN = GET_ROW(dataTable.normalData, k, dataTable.normalRowUnits);
        SET rowLN = GET_ROW(dataTable.normalData, l, dataTable.normalRowUnits);

        SET_INTERSECT(intersectionBuffer, rowIN, rowJN, normalBitsPerRow);
        SET_INTERSECT(intersectionBuffer, intersectionBuffer, rowKN,
                      normalBitsPerRow);
        SET_INTERSECT(intersectionBuffer, intersectionBuffer, rowLN,
                      normalBitsPerRow);

        int coveredNormal = SET_COUNT(intersectionBuffer, normalBitsPerRow);
        int TN = (int)dataTable.numNormal - coveredNormal;
        double F =
            (alpha * TP + TN) / (dataTable.numTumor + dataTable.numNormal);
        if (F >= maxF) {
          maxF = F;
          bestCombination = {computed.i, computed.j, k, l};
        }

        std::cout << "DEBUG: Combination (" << computed.i << ", " << computed.j
                  << ", " << k << ", " << l << ") maxF = " << maxF << ","
                  << " F: " << F << ", "
                  << "TP: " << TP << ", " << "TN: " << TN << ", "
                  << "numTumor: " << dataTable.numTumor << ", "
                  << "numNormal: " << dataTable.numNormal << "\n";
      }
    }
  }
}

bool process_and_communicate(int rank, LAMBDA_TYPE num_Comb,
                             double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb,
                             LAMBDA_TYPE &begin, LAMBDA_TYPE &end,
                             MPI_Status &status, sets_t dataTable,
                             SET &intersectionBuffer, SET &scratchBufferij,
                             SET &scratchBufferijk) {
  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          intersectionBuffer, scratchBufferij,
                          scratchBufferijk);

  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
  LAMBDA_TYPE next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);

  if (next_idx == -1) {
    return false;
  }

  begin = next_idx;
  end = std::min(begin + CHUNK_SIZE, num_Comb);
  return true;
}

void worker_process(int rank, LAMBDA_TYPE num_Comb, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb, sets_t dataTable,
                    SET &intersectionBuffer, SET &scratchBufferij,
                    SET &scratchBufferijk) {
  std::pair<LAMBDA_TYPE, LAMBDA_TYPE> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);

  LAMBDA_TYPE begin = chunk_indices.first;
  LAMBDA_TYPE end = chunk_indices.second;

  MPI_Status status;

  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, localBestMaxF, localComb, begin, end, status, dataTable,
        intersectionBuffer, scratchBufferij, scratchBufferijk);
    if (!has_next) {
      break;
    }
  }
}

void execute_role(int rank, int size_minus_one, LAMBDA_TYPE num_Comb,
                  double &localBestMaxF, std::array<int, NUMHITS> &localComb,
                  sets_t dataTable, SET &intersectionBuffer,
                  SET &scratchBufferij, SET &scratchBufferijk) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, localBestMaxF, localComb, dataTable,
                   intersectionBuffer, scratchBufferij, scratchBufferijk);
  }
}

std::array<int, NUMHITS> initialize_local_comb_and_f(double &f) {
  f = 0;
  std::array<int, NUMHITS> comb;
  comb.fill(-1);
  return comb;
}

MPIResultWithComb create_mpi_result(double f,
                                    const std::array<int, NUMHITS> &comb) {
  MPIResultWithComb result;
  result.f = f;
  for (int i = 0; i < NUMHITS; ++i) {
    result.comb[i] = comb[i];
  }
  return result;
}

std::array<int, NUMHITS>
extract_global_comb(const MPIResultWithComb &globalResult) {
  std::array<int, NUMHITS> globalBestComb;
  for (int i = 0; i < NUMHITS; ++i) {
    globalBestComb[i] = globalResult.comb[i];
  }
  return globalBestComb;
}

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable) {

  int Nt = dataTable.numTumor;
  int numGenes = dataTable.numRows;

  size_t tumorBits = dataTable.numTumor;
  size_t tumorUnits = dataTable.tumorRowUnits;
  size_t normalUnits = dataTable.normalRowUnits;
  size_t maxUnits = std::max(tumorUnits, normalUnits);

  SET intersectionBuffer, scratchBufferij, scratchBufferijk;
  SET_NEW(intersectionBuffer, tumorBits);
  SET_NEW(scratchBufferij, tumorBits);
  SET_NEW(scratchBufferijk, tumorBits);

  MPI_Datatype MPI_RESULT_WITH_COMB = create_mpi_result_with_comb_type();
  MPI_Op MPI_MAX_F_WITH_COMB = create_max_f_with_comb_op(MPI_RESULT_WITH_COMB);

  LAMBDA_TYPE num_Comb = nCr(numGenes, 2);

  SET droppedSamples;
  SET_NEW(droppedSamples, tumorBits);

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (!CHECK_ALL_BITS_SET(droppedSamples, tumorBits)) {
    double localBestMaxF;
    std::array<int, NUMHITS> localComb =
        initialize_local_comb_and_f(localBestMaxF);

    execute_role(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable,
                 intersectionBuffer, scratchBufferij, scratchBufferijk);

    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult;

    MPI_Allreduce(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                  MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);

    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);

    SET_COPY(intersectionBuffer,
             GET_ROW(dataTable.tumorData, globalBestComb[0], tumorUnits),
             tumorBits);

    for (int i = 1; i < NUMHITS; ++i)
      SET_INTERSECT(intersectionBuffer, intersectionBuffer,
                    GET_ROW(dataTable.tumorData, globalBestComb[i], tumorUnits),
                    tumorBits);

    SET_UNION(droppedSamples, droppedSamples, intersectionBuffer, tumorBits);
    UPDATE_SET_COLLECTION(dataTable.tumorData, intersectionBuffer,
                          dataTable.numRows, dataTable.tumorRowUnits);
    Nt -= SET_COUNT(intersectionBuffer, tumorBits);

    if (rank == 0)
      write_output(rank, outfile, globalBestComb, globalResult.f);
    break;
  }

  if (rank == 0)
    outfile.close();

  SET_FREE(intersectionBuffer);
  SET_FREE(scratchBufferij);
  SET_FREE(scratchBufferijk);
  SET_FREE(droppedSamples);

  MPI_Op_free(&MPI_MAX_F_WITH_COMB);
  MPI_Type_free(&MPI_RESULT_WITH_COMB);
}
