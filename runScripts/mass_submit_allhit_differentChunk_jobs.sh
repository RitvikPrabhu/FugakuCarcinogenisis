#!/bin/bash
#set -e

# Job parameters
NNODES=192
NPROC=$((NNODES * 48))
ELAPSE=${ELAPSE:-"01:00:00"}
GROUP=ra000012

# Define the cancer types; adjust if you have more than one.
cancers=("ACC")

# Map NUMHITS to the "hit name" part of the executable
declare -A hitNames
hitNames[2]="twoHit"
hitNames[3]="threeHit"
hitNames[5]="fiveHit"
hitNames[6]="sixHit"
hitNames[7]="sevenHit"
hitNames[8]="eightHit"
hitNames[9]="nineHit"

chunks_all=(1000 2000 3200 6400 12800 25600)

# List of NUMHITS values to create jobs for.
hitNumbers=(3)

for num in "${hitNumbers[@]}"; do
    chunkSizes=("${chunks_all[@]}")
    for chunk in "${chunkSizes[@]}"; do
      for cancer in "${cancers[@]}"; do
        JOB_NAME="hit_${num}_chunkSize_${chunk}"
        OUTPUT_DIR="batchSelection/${num}hit/${JOB_NAME}_out"

        # Construct the full executable name using hit name and chunk size.
        EXECUTABLE="../build/run_${hitNames[$num]}_${chunk}"

        PJSUB_ARGS=(
          -N "${JOB_NAME}"
          -g "${GROUP}"
          -o "${OUTPUT_DIR}/%j-${JOB_NAME}.stdout"
          -e "${OUTPUT_DIR}/%j-${JOB_NAME}.stderr"
          -L "node=${NNODES}"
          -L "elapse=${ELAPSE}"
          --mpi "proc=${NPROC}"
          -x "PJM_LLIO_GFSCACHE=/vol0004"
          -L "rscgrp=small"
          -m b,e
          --mail-list ritvikp@vt.edu
        )

        echo "Submitting job: ${JOB_NAME} for cancer ${cancer} using executable ${EXECUTABLE}"

        mkdir -p "${OUTPUT_DIR}"

        pjsub "${PJSUB_ARGS[@]}" << EOF
llio_transfer ${EXECUTABLE}
llio_transfer ../data/${cancer}.combinedData_sorted_bin.txt
mkdir -p "${OUTPUT_DIR}"
time mpirun ${EXECUTABLE} ../data/${cancer}.combinedData_sorted_bin.txt ${OUTPUT_DIR}/${chunk}_${num}hit.csv ${OUTPUT_DIR}/${chunk}_${num}hit.out
EOF
      done
    done
done

