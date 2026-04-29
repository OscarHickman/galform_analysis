#!/bin/bash
#SBATCH --job-name=hsamp_${MODE}_${SIM_NAME}_iz${IZ}
#SBATCH --output=logs/hsamp_${MODE}_${SIM_NAME}_iz${IZ}_%j.log
#SBATCH --error=logs/hsamp_${MODE}_${SIM_NAME}_iz${IZ}_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A durham
#SBATCH --partition=cosma5

set -euo pipefail

echo "======================================================================"
echo "HALO-SAMPLING CONVERGENCE: model=${MODEL_NAME:-lc16} mode=${MODE} sim=${SIM_NAME} iz=${IZ}"
echo "======================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "Start:   $(date)"
echo ""

cd /cosma/apps/durham/dc-hick2/galform_analysis
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${OUTPUT_DIR:-}}"
SUBVOLS_VALUE="${SUBVOLS:-}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SUBVOLS_VALUE="${SLURM_ARRAY_TASK_ID}"
    if [[ -n "${OUTPUT_DIR_BASE}" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/nsubvol_${SLURM_ARRAY_TASK_ID}"
    fi
fi

if [[ -z "${SUBVOLS_VALUE}" ]]; then
    echo "ERROR: No subvolume value provided (set SUBVOLS or use SLURM_ARRAY_TASK_ID)."
    exit 1
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "ERROR: OUTPUT_DIR/OUTPUT_DIR_BASE must be set."
    exit 1
fi

mkdir -p logs "${OUTPUT_DIR}"

module load python/3.9.19
export PYTHONPATH="$(pwd)/.venv/lib/python3.9/site-packages:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-lc16}"
DATA_ROOT="${DATA_ROOT:-/cosma5/data/durham/dc-hick2/Galform_TreeIDs}"
BASE_DIR="${BASE_DIR_OVERRIDE:-${DATA_ROOT}/${SIM_NAME}/${MODEL_NAME}}"

MSTAR_MIN_LOG10="${MSTAR_MIN_LOG10:-}"
CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
MHALO_MIN="${MHALO_MIN:-1e11}"
BOXSIZE="${BOXSIZE:-542.16}"

CMD=(
    python3 scripts/compute_halo_sampling_grid.py
    --base-dir "${BASE_DIR}"
    --sim-name "${SIM_NAME}"
    --iz "${IZ}"
    --subvols "${SUBVOLS_VALUE}"
    --nmax "${NMAX}"
    --output-dir "${OUTPUT_DIR}"
    --boxsize "${BOXSIZE}"
    --mhalo-min "${MHALO_MIN}"
    --num-threads "${SLURM_CPUS_PER_TASK}"
    --mode "${MODE}"
)

if [[ -n "${MSTAR_MIN_LOG10}" ]]; then
    CMD+=(--mstar-min-log10 "${MSTAR_MIN_LOG10}")
fi

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