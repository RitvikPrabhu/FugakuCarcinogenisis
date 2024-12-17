#!/bin/bash
# Dynamic node counts for multiple jobs

NODE_COUNTS=( 82944 )

for NODE_COUNT in "${NODE_COUNTS[@]}"; do
    RUN_SCRIPT="5hit_run_${NODE_COUNT}.sh"
	JOB_NAME="hit5_job_${NODE_COUNT}"
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
#PJM -g ra000012
#PJM -N ${JOB_NAME}
#PJM -L node=${NODE_COUNT}
#PJM -L elapse=02:00:00
#PJM --mpi proc=3981312
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=large"
#PJM -m b,e
#PJM --mail-list ritvikp@vt.edu

llio_transfer ../data/ACC.combinedData.txt
llio_transfer "bin/dataSparsity_5hit"
mpirun "bin/dataSparsity_5hit" ../data/ACC.combinedData.txt 5hit_metrics_${NODE_COUNT}.txt 5hit_${NODE_COUNT}.out

EOF
    chmod +x "$RUN_SCRIPT"
    pjsub "$RUN_SCRIPT"
done

