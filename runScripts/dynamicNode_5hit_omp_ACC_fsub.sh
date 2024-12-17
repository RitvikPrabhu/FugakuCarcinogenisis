#!/bin/bash
# Dynamic node counts for multiple jobs

NODE_COUNTS=( 82944 )

for NODE_COUNT in "${NODE_COUNTS[@]}"; do
    RUN_SCRIPT="5hit_run_${NODE_COUNT}_omp_ACC.sh"
	JOB_NAME="5hit_job_${NODE_COUNT}_omp_ACC"
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
llio_transfer ../data/ACC.combinedData.txt
llio_transfer "bin/dataSparsity_5hit_omp"
mpirun "bin/dataSparsity_5hit_omp" ../data/ACC.combinedData.txt 5hit_metrics_${NODE_COUNT}_omp_ACC.txt 5hit_${NODE_COUNT}_omp_ACC.out

EOF
    chmod +x "$RUN_SCRIPT"
    pjsub "$RUN_SCRIPT"
done

