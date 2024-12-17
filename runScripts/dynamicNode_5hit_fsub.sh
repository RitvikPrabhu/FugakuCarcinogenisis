#!/bin/bash
# Dynamic node counts for multiple jobs

NODE_COUNTS=( 5000 8000 10000 )

for NODE_COUNT in "${NODE_COUNTS[@]}"; do
    RUN_SCRIPT="run_${NODE_COUNT}.sh"
	JOB_NAME="job_check_${NODE_COUNT}_1ppn"
    cat <<EOF > "$RUN_SCRIPT"
#!/bin/bash
#PJM -g ra000012
#PJM -N ${JOB_NAME}
#PJM -L node=${NODE_COUNT}
#PJM -L elapse=02:00:00
#PJM --mpi proc=${NODE_COUNT}
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -L "rscgrp=large"
#PJM -m b,e
#PJM --mail-list ritvikp@vt.edu

llio_transfer ../data/ACC.combinedData.txt
llio_transfer "bin/dataSparsity_5hit"
mpirun "bin/dataSparsity_5hit" ../data/ACC.combinedData.txt metrics_${NODE_COUNT}.txt 4hit_${NODE_COUNT}.out

EOF
    chmod +x "$RUN_SCRIPT"
    pjsub "$RUN_SCRIPT"
done

