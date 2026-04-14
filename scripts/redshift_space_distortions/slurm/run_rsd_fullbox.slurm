#!/bin/bash
#SBATCH --job-name=rsd_fullbox
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -A durham
#SBATCH --partition=cosma5

set -euo pipefail

echo "======================================================================"
echo "RSD FULL-BOX STANDARD: sim=${SIM_NAME} model=${MODEL_NAME} iz=${IZ}"
echo "======================================================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "Start:   $(date)"
echo ""

cd /cosma/home/durham/dc-hick2/galform_analysis
mkdir -p logs "${OUTPUT_DIR}"

module load python/3.9.19
export PYTHONPATH="$(pwd)/.venv/lib/python3.9/site-packages:${PYTHONPATH:-}"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/${SIM_NAME}/${MODEL_NAME}}"
NMAX="${NMAX:-1024}"
BOXSIZE="${BOXSIZE:-542.16}"
MHALO_MIN="${MHALO_MIN:-1e10}"
N_RANDOM="${N_RANDOM:-200000}"
CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
MAX_GALAXIES_PER_IVOL="${MAX_GALAXIES_PER_IVOL:-0}"

CMD=(
    python3 scripts/redshift_space_distortions/compute_rsd_multipoles.py
    --base-dir "${BASE_DIR}"
    --sim-name "${SIM_NAME}"
    --model-name "${MODEL_NAME}"
    --iz "${IZ}"
    --nmax "${NMAX}"
    --boxsize "${BOXSIZE}"
    --mhalo-min "${MHALO_MIN}"
    --n-random "${N_RANDOM}"
    --max-galaxies-per-ivol "${MAX_GALAXIES_PER_IVOL}"
    --num-threads "${SLURM_CPUS_PER_TASK}"
    --output-dir "${OUTPUT_DIR}"
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
