#!/bin/bash
#set -x

#cancers=(
#  ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC
#  KICH KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV
#  PAAD PCPG PRAD READ SARC SKCM STAD TGCT THCA THYM
#  UCEC UCS UVM
#)

cancers=(
GBM OV UCEC
)

node_configs=(
  96
)

ELAPSE=${ELAPSE:-"03:00:00"}
GROUP=ra000012

for NNODES in "${node_configs[@]}"; do
  NPROC=$((NNODES * 48))

  for cancer in "${cancers[@]}"; do
    JOB_NAME="${cancer}_2hit_run_${NNODES}nodes"
    OUTPUT_DIR="strong_scaling/2hit/${cancer}/${NNODES}_nodes_out"

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
      --mail-list $(git config user.email)
    )

    echo "Submitting job for cancer: ${cancer} with ${NNODES} nodes"
    mkdir -p "${OUTPUT_DIR}"

    pjsub "${PJSUB_ARGS[@]}" << EOF
llio_transfer ../build/run_2hit
llio_transfer ../data/${cancer}.txt

mpiexec -mca common_tofu_num_mrq_entries 2097152 ../build/run_2hit \\
  ../data/${cancer}.txt \\
  $OUTPUT_DIR/${cancer}_2hit.csv \\
  $OUTPUT_DIR/${cancer}_2hit.out
EOF

  done
done
