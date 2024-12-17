#!/bin/bash
# Dynamic node counts for multiple jobs

NODE_COUNTS=( 82944 )

for NODE_COUNT in "${NODE_COUNTS[@]}"; do
    RUN_SCRIPT="6hit_run_${NODE_COUNT}_omp_BRCA.sh"
	JOB_NAME="hit6_job_${NODE_COUNT}_omp_BRCA"
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
#PJM -g ra000012
#PJM -N ${JOB_NAME}
#PJM -L node=${NODE_COUNT}
#PJM -L elapse=02:00:00
#PJM --mpi proc=${NODE_COUNT}
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=huge"
#PJM -m b,e
#PJM --mail-list ritvikp@vt.edu
export OMP_NUM_THREADS=48
llio_transfer ../data/BRCA.combinedData.txt
llio_transfer "bin/dataSparsity_6hit_omp"
mpirun "bin/dataSparsity_6hit_omp" ../data/BRCA.combinedData.txt 6hit_metrics_${NODE_COUNT}_omp_BRCA.txt 6hit_${NODE_COUNT}_omp_BRCA.out

EOF
    chmod +x "$RUN_SCRIPT"
    pjsub "$RUN_SCRIPT"
done
