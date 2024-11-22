#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_data_file> <metrics_data_file> <results_file> "
    exit 1
fi

COMBINED_DATA_FILE="$1"
METRICS_OUTPUT_FILE="$2"
RESULTS_FILE="$3"


GENE_SAMPLE_BASENAME=$(basename "$COMBINED_DATA_FILE")
GENE_SAMPLE_PREFIX=${GENE_SAMPLE_BASENAME%%.*}       
PRUNED_DATA_FILE="3hit_pruned_${GENE_SAMPLE_PREFIX}_data.bin"

#mpiFCC -Nclang -std=c++11 -Ofast -o ../sparsification/dataSparsity_3hit ../sparsification/dataSparsity_3hit.cpp 
mpiFCC -Nclang -std=c++11 -Ofast -Kopenmp -o ../setcover/3hitCombination ../setcover/3hitCombination.cpp 


#mpirun ../sparsification/dataSparsity_3hit "../data/${COMBINED_DATA_FILE}" $METRICS_OUTPUT_FILE "../data/${PRUNED_DATA_FILE}"
mpirun ../setcover/3hitCombination "../data/${PRUNED_DATA_FILE}" "../data/${COMBINED_DATA_FILE}" $METRICS_OUTPUT_FILE $RESULTS_FILE




rm ../setcover/3hitCombination
#rm ../sparsification/dataSparsity_3hit
