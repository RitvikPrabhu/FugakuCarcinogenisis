#!/usr/bin/bash

# COMPUTE NODE$ ./fapp_run.sh
set -x

FAPPOUT=fapp_results.${PJM_JOBID}
mkdir -p "${FAPPOUT}"

for i in $(seq 1 17); do 
  fapp -C -d "${FAPPOUT}/rep${i}" -Icpupa,mpi -Hevent="pa${i}" mpirun ../build/run ../data/ACC.combinedData_sorted_bin.txt metrics_4hit.txt 4hitTrial_4node.out
done
