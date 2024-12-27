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

#include "constants.h"
#include "four_hit.h"
#include "mpi_specific.h"
#include "utils.h"

LambdaComputed compute_lambda_variables(long long int lambda, int totalGenes) {
  LambdaComputed computed;
  computed.j = static_cast<int>(std::floor(std::sqrt(0.25 + 2 * lambda) + 0.5));
  if (computed.j > totalGenes - (NUMHITS - 2)) {
    computed.j = -1;
    return computed;
  }
  computed.i = lambda - (computed.j * (computed.j - 1)) / 2;
  return computed;
}

void update_best_combination(
    double &globalMaxF, std::array<int, NUMHITS> &globalBestCombination,
    double localMaxF, const std::array<int, NUMHITS> &localBestCombination) {
  if (localMaxF >= globalMaxF) {
    globalMaxF = localMaxF;
    globalBestCombination = localBestCombination;
  }
}

void execute_role(int rank, int size_minus_one, long long int num_Comb,
                  unsigned long long **&tumorData,
                  unsigned long long **&normalData, int numGenes, int Nt,
                  int Nn, double &localBestMaxF,
                  std::array<int, NUMHITS> &localComb) {
  if (rank == 0) {
    master_process(size_minus_one, num_Comb);
  } else {
    worker_process(rank, num_Comb, tumorData, normalData, numGenes, Nt, Nn,
                   localBestMaxF, localComb);
  }
}

void write_output(int rank, std::ofstream &outfile,
                  const std::array<int, NUMHITS> &globalBestComb,
                  const std::string *geneIdArray, double F_max) {
  outfile << "(";
  for (size_t idx = 0; idx < globalBestComb.size(); ++idx) {
    outfile << geneIdArray[globalBestComb[idx]];
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

MPIResultWithComb
perform_MPI_allreduce_with_comb(const MPIResultWithComb &localResult,
                                MPI_Op MPI_MAX_FSCORE_WITH_COMB,
                                MPI_Datatype MPI_RESULT_WITH_COMB) {
  MPIResultWithComb globalResult;
  MPI_Allreduce(&localResult, &globalResult, 1, MPI_RESULT_WITH_COMB,
                MPI_MAX_FSCORE_WITH_COMB, MPI_COMM_WORLD);
  return globalResult;
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

void process_lambda_interval(unsigned long long **&tumorData,
                             unsigned long long **&normalData,
                             long long int startComb, long long int endComb,
                             int totalGenes,
                             std::array<int, NUMHITS> &bestCombination, int Nt,
                             int Nn, double &maxF) {
  const double alpha = 0.1;

#pragma omp parallel
  {
    double localMaxF = maxF;
    std::array<int, NUMHITS> localBestCombination = bestCombination;

#pragma omp for nowait schedule(dynamic)
    for (long long int lambda = startComb; lambda <= endComb; lambda++) {
      LambdaComputed computed = compute_lambda_variables(lambda, totalGenes);
      if (computed.j == -1)
        continue;

      unsigned long long *intersectTumor1 =
          get_intersection(tumorData, Nt, computed.i, computed.j, -1);

      if (is_empty(intersectTumor1, calculate_bit_units(Nt)))
        continue;

      for (int k = computed.j + 1; k < totalGenes - (NUMHITS - 3); k++) {
        unsigned long long *intersectTumor2 =
            get_intersection(tumorData, Nt, computed.i, computed.j, k, -1);

        if (is_empty(intersectTumor2, calculate_bit_units(Nt)))
          continue;

        for (int l = k + 1; l < totalGenes - (NUMHITS - 4); l++) {
          unsigned long long *intersectTumor3 =
              get_intersection(tumorData, Nt, computed.i, computed.j, k, l, -1);

          if (is_empty(intersectTumor3, calculate_bit_units(Nt)))
            continue;

          unsigned long long *intersectNormal = get_intersection(
              normalData, Nn, computed.i, computed.j, k, l, -1);

          int TP = bitCollection_size(intersectTumor3, calculate_bit_units(Nt));
          int TN =
              Nn - bitCollection_size(intersectNormal, calculate_bit_units(Nn));

          double F = compute_F(TP, TN, alpha, Nt, Nn);
          if (F >= localMaxF) {
            localMaxF = F;
            localBestCombination = {computed.i, computed.j, k, l};
          }
        }
      }
    }

#pragma omp critical
    {
      update_best_combination(maxF, bestCombination, localMaxF,
                              localBestCombination);
    }
  }
}

bool process_and_communicate(int rank, long long int num_Comb,
                             unsigned long long **&tumorData,
                             unsigned long long **&normalData, int numGenes,
                             int Nt, int Nn, double &localBestMaxF,
                             std::array<int, NUMHITS> &localComb,
                             long long int &begin, long long int &end,
                             MPI_Status &status) {

  process_lambda_interval(tumorData, normalData, begin, end, numGenes,
                          localComb, Nt, Nn, localBestMaxF);
  notify_master_chunk_processed(0, 1);
  long long int next_idx = receive_next_chunk_index(status, 0, 2);

  if (next_idx == -1)
    return false;

  begin = next_idx;
  end = std::min(next_idx + CHUNK_SIZE, num_Comb);
  return true;
}

void worker_process(int rank, long long int num_Comb,
                    unsigned long long **&tumorData,
                    unsigned long long **&normalData, int numGenes, int Nt,
                    int Nn, double &localBestMaxF,
                    std::array<int, NUMHITS> &localComb) {

  std::pair<long long int, long long int> chunk_indices =
      calculate_initial_chunk(rank, num_Comb, CHUNK_SIZE);
  long long int begin = chunk_indices.first;
  long long int end = chunk_indices.second;
  MPI_Status status;
  while (end <= num_Comb) {
    bool has_next = process_and_communicate(
        rank, num_Comb, tumorData, normalData, numGenes, Nt, Nn, localBestMaxF,
        localComb, begin, end, status);
    if (!has_next)
      break;
  }
}

std::array<int, NUMHITS> initialize_local_comb_and_f(double &f) {
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

void distribute_tasks(int rank, int size, int numGenes,
                      unsigned long long **&tumorData,
                      unsigned long long **&normalData, int Nt, int Nn,
                      const char *outFilename,
                      unsigned long long *&tumorSamples,
                      std::string *geneIdArray, double elapsed_times[]) {

  MPI_Datatype MPI_RESULT_WITH_COMB = create_mpi_result_with_comb_type();
  MPI_Op MPI_MAX_F_WITH_COMB = create_max_f_with_comb_op(MPI_RESULT_WITH_COMB);
  long long int num_Comb = nCr(numGenes, 2);
  double master_worker_time = 0, all_reduce_time = 0, broadcast_time = 0;
  unsigned long long *droppedSamples =
      initialize_dropped_samples(calculate_bit_units(Nt));

  std::ofstream outfile;
  if (rank == 0) {
    outfile.open(outFilename);
    outputFileWriteError(outfile);
  }

  while (!arrays_equal(tumorSamples, droppedSamples, calculate_bit_units(Nt))) {
    double localBestMaxF;
    std::array<int, NUMHITS> localComb =
        initialize_local_comb_and_f(localBestMaxF);

    START_TIMING(master_worker)
    execute_role(rank, size - 1, num_Comb, tumorData, normalData, numGenes, Nt,
                 Nn, localBestMaxF, localComb);
    END_TIMING(master_worker, master_worker_time);

    START_TIMING(all_reduce)
    MPIResultWithComb localResult = create_mpi_result(localBestMaxF, localComb);
    MPIResultWithComb globalResult = perform_MPI_allreduce_with_comb(
        localResult, MPI_MAX_F_WITH_COMB, MPI_RESULT_WITH_COMB);
    std::array<int, NUMHITS> globalBestComb = extract_global_comb(globalResult);
    END_TIMING(all_reduce, all_reduce_time);

    unsigned long long *sampleToCover =
        get_intersection(tumorData, Nt, globalBestComb[0], globalBestComb[1],
                         globalBestComb[2], globalBestComb[3], -1);
    update_dropped_samples(droppedSamples, sampleToCover,
                           calculate_bit_units(Nt));

    update_tumor_data(tumorData, sampleToCover, calculate_bit_units(Nt),
                      numGenes);

    updateNt(Nt, sampleToCover);

    if (rank == 0) {
      write_output(rank, outfile, globalBestComb, geneIdArray, globalResult.f);
    }
    delete[] sampleToCover;
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
