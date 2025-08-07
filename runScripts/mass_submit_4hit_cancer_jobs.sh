#!/bin/bash
set -x

#cancers=(
#  ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC
#  KICH KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV
#  PAAD PCPG PRAD READ SARC SKCM STAD TGCT THCA THYM
#  UCEC UCS UVM
#)

cancers=(
ACC.combinedData_sorted_bin
)

node_configs=(
  400
)

ELAPSE=${ELAPSE:-"00:10:00"}
GROUP=ra000012

for NNODES in "${node_configs[@]}"; do
  NPROC=$((NNODES * 48))

  for cancer in "${cancers[@]}"; do
    JOB_NAME="${cancer}_4hit_run_${NNODES}nodes"
    OUTPUT_DIR="strong_scaling/4hit/${cancer}/${NNODES}_nodes_out"

    PJSUB_ARGS=(
      -N "${JOB_NAME}"
      -g "${GROUP}"
      -o "${OUTPUT_DIR}/%j-${JOB_NAME}.stdout"
      -e "${OUTPUT_DIR}/%j-${JOB_NAME}.stderr"
      -L "node=${NNODES}"
      -L "elapse=${ELAPSE}"
      --mpi "proc=${NPROC}"
      -x "PJM_LLIO_GFSCACHE=/vol0004"
      -L "rscgrp=large"
      -m b,e
      #--mail-list $(git config user.email)
      --mail-list ritvikp@vt.edu
    )

    echo "Submitting job for cancer: ${cancer} with ${NNODES} nodes"
    mkdir -p "${OUTPUT_DIR}"

    pjsub "${PJSUB_ARGS[@]}" << EOF
llio_transfer ../build/run_4hit
llio_transfer ../data/${cancer}.txt

mpiexec ../build/run_4hit \\
  ../data/${cancer}.txt \\
  $OUTPUT_DIR/${cancer}_4hit_profile.csv \\
  $OUTPUT_DIR/${cancer}_4hit_profile.out
EOF

  done
done
