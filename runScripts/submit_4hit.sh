#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_data_file> <metrics_data_file> <results_file> <node_count>"
    exit 1
fi

COMBINED_DATA_FILE="$1"
METRICS_OUTPUT_FILE="$2"
RESULTS_FILE="$3"
NODE_COUNT="$4"

GENE_SAMPLE_BASENAME=$(basename "$COMBINED_DATA_FILE")
GENE_SAMPLE_PREFIX=${GENE_SAMPLE_BASENAME%%.*}
PRUNED_DATA_FILE="4hit_pruned_${GENE_SAMPLE_PREFIX}_data_${NODE_COUNT}.bin"
PRUNED_3HIT_FILE="3hit_pruned_${GENE_SAMPLE_PREFIX}_data_${NODE_COUNT}.bin"

BIN_DIR="bin_${NODE_COUNT}"
mkdir -p "$BIN_DIR"

mpiFCC -Nclang -std=c++11 -Ofast -fopenmp -o "$BIN_DIR/dataSparsity_4hit" ../sparsification/dataSparsity_4hit.cpp
llio_transfer "$BIN_DIR/dataSparsity_4hit" 
export OMP_NUM_THREADS=48
mpirun "$BIN_DIR/dataSparsity_4hit" "../data/${COMBINED_DATA_FILE}" "../data/${PRUNED_3HIT_FILE}" "4hit_${NODE_COUNT}.out"

# Clean up binaries for this node count
rm -r "$BIN_DIR"

