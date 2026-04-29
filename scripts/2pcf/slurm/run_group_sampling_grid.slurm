#!/bin/bash
#SBATCH --job-name=gsamp_job
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A durham
#SBATCH --partition=cosma5

set -euo pipefail

echo "======================================================================"
echo "GROUP-SAMPLING CONVERGENCE: model=${MODEL_NAME:-gp14} mode=${MODE} sim=${SIM_NAME} iz=${IZ} n_subvol=${N_SUBVOL}"
echo "======================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "Start:   $(date)"
echo ""

cd /cosma/apps/durham/dc-hick2/galform_analysis
mkdir -p logs "${OUTPUT_DIR}"

module load python/3.9.19
export PYTHONPATH="$(pwd)/.venv/lib/python3.9/site-packages:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-gp14}"
BASE_DIR="/cosma5/data/durham/dc-hick2/Galform_Out/${SIM_NAME}/${MODEL_NAME}"

MSTAR_MIN_LOG10="${MSTAR_MIN_LOG10:-9.0}"
CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
MHALO_MIN="${MHALO_MIN:-1e11}"
BOXSIZE="${BOXSIZE:-542.16}"

CMD=(
    python3 scripts/2pcf/compute_group_sampling_grid.py
    --base-dir "${BASE_DIR}"
    --sim-name "${SIM_NAME}"
    --iz "${IZ}"
    --subvols "${N_SUBVOL}"
    --nmax "${NMAX}"
    --output-dir "${OUTPUT_DIR}"
    --boxsize "${BOXSIZE}"
    --mstar-min-log10 "${MSTAR_MIN_LOG10}"
    --mhalo-min "${MHALO_MIN}"
    --num-threads "${SLURM_CPUS_PER_TASK}"
    --mode "${MODE}"
)

if [[ "${CENTRALS_ONLY}" == "1" ]]; then
    CMD+=(--centrals-only)
fi

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

EXIT_CODE=$?
echo ""
echo "End: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================================================"
exit ${EXIT_CODE}
