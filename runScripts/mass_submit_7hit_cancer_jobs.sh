#!/bin/bash
#set -x

#cancers=(
#  ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC
#  KICH KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV
#  PAAD PCPG PRAD READ SARC SKCM STAD TGCT THCA THYM
#  UCEC UCS UVM
#)

cancers=(
DLBC ESCA PRAD
)

node_configs=(
  6144
)

ELAPSE=${ELAPSE:-"24:00:00"}
GROUP=ra000012

for NNODES in "${node_configs[@]}"; do
  NPROC=$((NNODES * 48))

  for cancer in "${cancers[@]}"; do
    JOB_NAME="${cancer}_7hit_run_${NNODES}nodes"
    OUTPUT_DIR="strong_scaling/7hit/${cancer}/${NNODES}_nodes_out"

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
      --mail-list $(git config user.email)
    )

    echo "Submitting job for cancer: ${cancer} with ${NNODES} nodes"
    mkdir -p "${OUTPUT_DIR}"

    pjsub "${PJSUB_ARGS[@]}" << EOF
llio_transfer ../build/run_7hit
llio_transfer ../data/${cancer}.txt

mpiexec -mca common_tofu_num_mrq_entries 2097152 ../build/run_7hit \\
  ../data/${cancer}.txt \\
  $OUTPUT_DIR/${cancer}_7hit.csv \\
  $OUTPUT_DIR/${cancer}_7hit.out
EOF

  done
done
