#!/usr/bin/bash

# LOGIN NODE$ fapp_package.sh ./fapp_results.<<<JOBID>>>

set -x

FAPPOUT=$1
OUTDIR=${FAPPOUT}/output
mkdir -p "${OUTDIR}"

for i in $(seq 1 17); do 
  fapppx -A -d "${FAPPOUT}/rep${i}" -Icpupa,mpi -tcsv -o "${OUTDIR}/pa${i}.csv"
done

cp "$(dirname "$(which fccpx)")/../misc/cpupa/cpu_pa_report.xlsm" "${OUTDIR}"
# rm -rf tmp
rm ~/fapp_out.zip
zip -r ~/fapp_out.zip "${OUTDIR}"
