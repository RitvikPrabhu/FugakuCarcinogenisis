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

SET calculate_initial_index(int num_workers) {
  return num_workers * CHUNK_SIZE;
}

void distribute_work(int num_workers, SET num_Comb, SET &next_idx) {

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

void master_process(int num_workers, SET num_Comb) {
  SET next_idx = calculate_initial_index(num_workers);
  distribute_work(num_workers, num_Comb, next_idx);
  SET termination_signal = -1;
  for (int workerRank = 1; workerRank <= num_workers; ++workerRank) {
    MPI_Send(&termination_signal, 1, MPI_LONG_LONG_INT, workerRank, 2,
             MPI_COMM_WORLD);
  }
}

inline SET nCr(int n, int r) {
  if (r > n)
    return 0;
  if (r == 0 || r == n)
    return 1;
  if (r > n - r)
    r = n - r; // Because C(n, r) == C(n, n-r)

  SET result = 1;
  for (int i = 1; i <= r; ++i) {
    result *= (n - r + i);
    result /= i;
  }
  return result;
}

bool arrays_equal(const SET_COLLECTION a, const SET_COLLECTION b,
                  size_t units) {
  for (size_t i = 0; i < units; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

double compute_F(int TP, int TN, double alpha, int Nt, int Nn) {
  return (alpha * TP + TN) / (Nt + Nn);
}

void update_tumor_data(SET_COLLECTION &tumorData, SET_COLLECTION sampleToCover,
                       size_t units, int numGenes) {
  for (int gene = 0; gene < numGenes; ++gene) {
    SET_COLLECTION geneRow = tumorData + gene * units;
    for (size_t i = 0; i < units; ++i) {
      geneRow[i] &= ~sampleToCover[i];
    }
  }
}

void outputFileWriteError(std::ofstream &outfile) {

  if (!outfile) {
    std::cerr << "Error: Could not open output file." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

std::pair<SET, SET> calculate_initial_chunk(int rank, SET num_Comb,
                                            SET chunk_size) {
  SET begin = (rank - 1) * chunk_size;
  SET end = std::min(begin + chunk_size, num_Comb);
  return {begin, end};
}

void update_dropped_samples(SET_COLLECTION &droppedSamples,
                            SET_COLLECTION sampleToCover, size_t units) {
  for (size_t i = 0; i < units; ++i) {
    droppedSamples[i] |= sampleToCover[i];
  }
}

SET_COLLECTION initialize_dropped_samples(size_t units) {
  SET_COLLECTION droppedSamples = new SET[units];
  memset(droppedSamples, 0, units * sizeof(SET));
  return droppedSamples;
}

LambdaComputed compute_lambda_variables(SET lambda, int totalGenes) {
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

void process_lambda_interval(SET startComb, SET endComb,
                             std::array<int, NUMHITS> &bestCombination,
                             double &maxF, sets_t dataTable,
                             SET_COLLECTION &intersectionBuffer,
                             SET_COLLECTION &scratchBufferij,
                             SET_COLLECTION &scratchBufferijk) {
  double alpha = 0.1;
  size_t tumorUnits = UNITS_FOR_BITS(dataTable.numTumor);
  size_t normalUnits = UNITS_FOR_BITS(dataTable.numNormal);
  int totalGenes = dataTable.numRows;

  for (SET lambda = startComb; lambda <= endComb; lambda++) {
    LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);

    SET_COLLECTION rowI = dataTable.tumorData + computed.i * tumorUnits;
    SET_COLLECTION rowJ = dataTable.tumorData + computed.j * tumorUnits;

    intersect_two_rows(scratchBufferij, rowI, rowJ, tumorUnits);

    if (is_empty(scratchBufferij, dataTable.numTumor))
      continue;

    for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {

      SET_COLLECTION rowK = dataTable.tumorData + k * tumorUnits;
      intersect_two_rows(scratchBufferijk, scratchBufferij, rowK, tumorUnits);

      if (is_empty(scratchBufferijk, dataTable.numTumor))
        continue;

      for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
        SET_COLLECTION rowL = dataTable.tumorData + l * tumorUnits;
        intersect_two_rows(intersectionBuffer, scratchBufferijk, rowL,
                           tumorUnits);

        if (is_empty(intersectionBuffer, dataTable.numTumor))
          continue;

        int TP = bitCollection_size(intersectionBuffer, dataTable.numTumor);

        SET_COLLECTION rowIN = dataTable.normalData + computed.i * normalUnits;
        SET_COLLECTION rowJN = dataTable.normalData + computed.j * normalUnits;
        SET_COLLECTION rowKN = dataTable.normalData + k * normalUnits;
        SET_COLLECTION rowLN = dataTable.normalData + l * normalUnits;

        intersect_two_rows(intersectionBuffer, rowIN, rowJN, normalUnits);
        intersect_two_rows(intersectionBuffer, intersectionBuffer, rowKN,
                           normalUnits);
        intersect_two_rows(intersectionBuffer, intersectionBuffer, rowLN,
                           normalUnits);

        /*
        for (size_t b = 0; b < normalUnits; b++) {
          intersectionBuffer[b] = rowIN[b] & rowJN[b] & rowKN[b] & rowLN[b];
        }*/

        int coveredNormal =
            bitCollection_size(intersectionBuffer, dataTable.numNormal);
        int TN = dataTable.numNormal - coveredNormal;
        double F =
            (alpha * TP + TN) / (dataTable.numTumor + dataTable.numNormal);
        if (F >= maxF) {
          maxF = F;
          bestCombination = {computed.i, computed.j, k, l};
        }
      }
    }
  }
}

bool process_and_communicate(int rank, SET num_Comb, double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb, SET &begin,
                             SET &end, MPI_Status &status, sets_t dataTable,
                             SET_COLLECTION &intersectionBuffer,
                             SET_COLLECTION &scratchBufferij,
                             SET_COLLECTION &scratchBufferijk) {

  process_lambda_interval(begin, end, localComb, localBestMaxF, dataTable,
                          intersectionBuffer, scratchBufferij,
                          scratchBufferijk);
  char signal = 'a';
  MPI_Send(&signal, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

  SET next_idx;
  MPI_Recv(&next_idx, 1, MPI_LONG_LONG_INT, 0, 2, MPI_COMM_WORLD, &status);

  if (next_idx == -1)
    return false;

  begin = next_idx;
  end = std::min(next_idx + CHUNK_SIZE, num_Comb);
  return true;
}

void worker_process(int rank, SET num_Comb, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb, sets_t dataTable,
                    SET_COLLECTION &intersectionBuffer,
                    SET_COLLECTION &scratchBufferij,
                    SET_COLLECTION &scratchBufferijk) {

  std::pair<SET, SET> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);
  SET begin = chunk_indices.first;
  SET end = chunk_indices.second;
  MPI_Status status;
  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, localBestMaxF, localComb, begin, end, status, dataTable,
        intersectionBuffer, scratchBufferij, scratchBufferijk);
    if (!has_next)
      break;
  }
}

void execute_role(int rank, int size_minus_one, SET num_Comb,
                  double &localBestMaxF, std::array<int, NUMHITS> &localComb,
                  sets_t dataTable, SET_COLLECTION &intersectionBuffer,
                  SET_COLLECTION &scratchBufferij,
                  SET_COLLECTION &scratchBufferijk) {
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

bool all_bits_set(const SET_COLLECTION droppedSamples, size_t units) {
  return std::all_of(droppedSamples, droppedSamples + units,
                     [](SET val) { return val == ~static_cast<SET>(0); });
}

void distribute_tasks(int rank, int size, const char *outFilename,
                      double elapsed_times[], sets_t dataTable) {

  int Nt = dataTable.numTumor;
  int Nn = dataTable.numNormal;
  int numGenes = dataTable.numRows;

  size_t tumorUnits = UNITS_FOR_BITS(Nt);
  size_t normalUnits = UNITS_FOR_BITS(Nn);
  size_t maxUnits = std::max(tumorUnits, normalUnits);

  SET_COLLECTION intersectionBuffer = new SET[maxUnits];
  SET_COLLECTION scratchBufferij = new SET[maxUnits];
  SET_COLLECTION scratchBufferijk = new SET[maxUnits];

  MPI_Datatype MPI_RESULT_WITH_COMB = create_mpi_result_with_comb_type();
  MPI_Op MPI_MAX_F_WITH_COMB = create_max_f_with_comb_op(MPI_RESULT_WITH_COMB);
  SET num_Comb = nCr(numGenes, 2);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  SET_COLLECTION droppedSamples =
      initialize_dropped_samples(CALCULATE_BIT_UNITS(Nt));

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (!all_bits_set(droppedSamples, CALCULATE_BIT_UNITS(Nt))) {
    double localBestMaxF;
    std::array<int, NUMHITS> localComb =
        initialize_local_comb_and_f(localBestMaxF);

    START_TIMING(master_worker)
    execute_role(rank, size - 1, num_Comb, localBestMaxF, localComb, dataTable,
                 intersectionBuffer, scratchBufferij, scratchBufferijk);
    END_TIMING(master_worker, master_worker_time);

    START_TIMING(all_reduce)
    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult;
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                  MPI_MAX_F_WITH_COMB, MPI_COMM_WORLD);

    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);
    END_TIMING(all_reduce, all_reduce_time);

    load_first_tumor(intersectionBuffer, dataTable, globalBestComb[0]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[1]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[2]);
    inplace_intersect_tumor(intersectionBuffer, dataTable, globalBestComb[3]);

    update_dropped_samples(droppedSamples, intersectionBuffer, tumorUnits);

    update_tumor_data(dataTable.tumorData, intersectionBuffer, tumorUnits,
                      numGenes);

    Nt -= bitCollection_size(intersectionBuffer, dataTable.numTumor);

    if (rank == 0) {
      write_output(rank, outfile, globalBestComb, globalResult.f);
    }
    // break;
  }
  if (rank == 0) {
    outfile.close();
  }

  elapsed_times[MASTER_WORKER] = master_worker_time;
  elapsed_times[ALL_REDUCE] = all_reduce_time;
  elapsed_times[BCAST] = broadcast_time;

  delete[] droppedSamples;
  MPI_Op_free(&MPI_MAX_F_WITH_COMB);
  MPI_Type_free(&MPI_RESULT_WITH_COMB);
}
