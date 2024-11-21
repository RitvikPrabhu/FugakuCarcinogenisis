#!/bin/bash
#PJM -g ra000012
#PJM -L node=4
#PJM -L elapse=01:00:00
#PJM --mpi proc=4
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=small"

export OMP_NUM_THREADS=48
./run.sh
