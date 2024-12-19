#!/bin/bash

mpirun -np 3 sparsification/dataSparsity_4hit_omp data/small_ACC.combinedData_sorted.txt 4hit_metrics_omp_ACC.txt 4hit_omp_ACC.out
