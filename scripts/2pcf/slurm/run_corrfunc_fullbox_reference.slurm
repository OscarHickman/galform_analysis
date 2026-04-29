#!/bin/bash
#SBATCH --job-name=cfull_${SIM_NAME}_iz${IZ}
#SBATCH --output=logs/cfull_${SIM_NAME}_iz${IZ}_%j.log
#SBATCH --error=logs/cfull_${SIM_NAME}_iz${IZ}_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A durham
#SBATCH --partition=cosma5

set -euo pipefail

echo "======================================================================"
echo "CORRFUNC FULLBOX REFERENCE: model=${MODEL_NAME:-lc16} sim=${SIM_NAME} iz=${IZ}"
echo "======================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "Start:   $(date)"
echo ""

cd /cosma/apps/durham/dc-hick2/galform_analysis

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "ERROR: OUTPUT_DIR must be set."
    exit 1
fi

mkdir -p logs "${OUTPUT_DIR}"

module load python/3.9.19
export PYTHONPATH="$(pwd)/.venv/lib/python3.9/site-packages:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-lc16}"
DATA_ROOT="${DATA_ROOT:-/cosma5/data/durham/dc-hick2/Galform_TreeIDs}"
BASE_DIR="${BASE_DIR_OVERRIDE:-${DATA_ROOT}/${SIM_NAME}/${MODEL_NAME}}"

CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
MHALO_MIN="${MHALO_MIN:-1e11}"
N_SUBVOL="${N_SUBVOL:-1024}"
IVOL_START="${IVOL_START:-0}"

CMD=(
    python3 scripts/compute_corrfunc_fullbox_reference.py
    --base-dir "${BASE_DIR}"
    --sim-name "${SIM_NAME}"
    --iz "${IZ}"
    --n-subvol "${N_SUBVOL}"
    --ivol-start "${IVOL_START}"
    --output-dir "${OUTPUT_DIR}"
    --mhalo-min "${MHALO_MIN}"
    --num-threads "${SLURM_CPUS_PER_TASK}"
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
