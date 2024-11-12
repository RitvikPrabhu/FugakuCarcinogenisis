#!/bin/bash
#PJM -g ra000012
#PJM -L node=2
#PJM -L elapse=01:00:00
#PJM --mpi proc=96
#PJM -x PJM_LLIO_GFSCACHE=/vol0004


mpiFCC -Nclang -std=c++11 -Ofast -o graphSparsity_3hit_mpi graphSparsity_3hit_mpi.cpp
mpirun ./graphSparsity_3hit_mpi smallInput_5genes.txt
find . -type f -name "combinations_rank_*" -size +0 -print