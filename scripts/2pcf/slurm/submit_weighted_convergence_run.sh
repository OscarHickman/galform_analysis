#!/bin/bash
#SBATCH --job-name=wconv_${IZ_NUM}_${MASS_TAG}_${GAL_TAG}
#SBATCH --output=logs/wconv_${IZ_NUM}_${MASS_TAG}_${GAL_TAG}_%j.log
#SBATCH --error=logs/wconv_${IZ_NUM}_${MASS_TAG}_${GAL_TAG}_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A durham
#SBATCH --partition=cosma5

echo "======================================================================"
echo "WEIGHTED 2PCF CONVERGENCE: iz${IZ_NUM}  mhalo>${MHALO_MIN}  gals=${GAL_TAG}"
echo "======================================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo ""

cd /cosma/apps/durham/dc-hick2/galform_analysis

module load python/3.9.19
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

# Build satellite flag
SAT_FLAG=""
if [ "${INCLUDE_SATELLITES}" = "1" ]; then
    SAT_FLAG="--include-satellites"
fi

CMD="python3 scripts/compute_weighted_convergence_specific.py \
    --iz ${IZ_NUM} \
    --subvols 1024 \
    --output-dir \"${OUTPUT_DIR}\" \
    --mhalo-min ${MHALO_MIN} \
    ${SAT_FLAG}"

echo "Running: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"
exit $EXIT_CODE
