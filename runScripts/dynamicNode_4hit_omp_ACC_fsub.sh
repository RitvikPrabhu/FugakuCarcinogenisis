#!/bin/bash
# Dynamic node counts for multiple jobs

NODE_COUNTS=( 3 )

for NODE_COUNT in "${NODE_COUNTS[@]}"; do
    RUN_SCRIPT="4hit_run_${NODE_COUNT}_omp_ACC.sh"
	JOB_NAME="hit4_job_${NODE_COUNT}_omp_ACC"
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
#PJM -g ra000012
#PJM -N ${JOB_NAME}
#PJM -L node=${NODE_COUNT}
#PJM -L elapse=00:10:00
#PJM --mpi proc=${NODE_COUNT}
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=small"
#PJM -m b,e
#PJM --mail-list ritvikp@vt.edu
export OMP_NUM_THREADS=48
llio_transfer ../data/small_ACC.combinedData_sorted.txt
llio_transfer "bin/a.out"
mpirun "bin/a.out" ../data/small_ACC.combinedData_sorted.txt 4hit_metrics_${NODE_COUNT}_omp_ACC.txt 4hit_${NODE_COUNT}_omp_ACC.out

EOF
    chmod +x "$RUN_SCRIPT"
    pjsub "$RUN_SCRIPT"
done

