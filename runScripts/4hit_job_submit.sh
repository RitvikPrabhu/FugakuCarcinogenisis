#!/bin/bash
#PJM -g ra000012
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM --mpi proc=48
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=small"
#PJM -m b,e
#PJM --mail-list ritvikp@vt.edu
mpirun ../build/run ../data/ACC.combinedData_sorted_bin.txt metrics_4hit.txt 4hitTrial_4node.out

