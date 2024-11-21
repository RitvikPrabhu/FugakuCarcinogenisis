#!/bin/bash
#PJM -g ra000012
#PJM -L node=9
#PJM -L elapse=01:00:00
#PJM --mpi proc=432
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=small"
./run.sh
