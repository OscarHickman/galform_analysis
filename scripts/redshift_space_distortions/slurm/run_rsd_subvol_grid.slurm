#!/bin/bash
#SBATCH --job-name=rsd_subvol
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -A durham
#SBATCH --partition=cosma5

set -euo pipefail

cd /cosma/home/durham/dc-hick2/galform_analysis
mkdir -p logs "${OUTPUT_DIR}"

module load python/3.9.19
export PYTHONPATH="$(pwd)/.venv/lib/python3.9/site-packages:${PYTHONPATH:-}"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/${SIM_NAME}/${MODEL_NAME}}"

CMD=(
  python3 scripts/redshift_space_distortions/compute_rsd_subvol_grid.py
  --base-dir "${BASE_DIR}"
  --sim-name "${SIM_NAME}"
  --model-name "${MODEL_NAME}"
  --iz "${IZ}"
  --n-subvol "${N_SUBVOL}"
  --nmax "${NMAX}"
  --mode "${MODE}"
  --boxsize "${BOXSIZE}"
  --mhalo-min "${MHALO_MIN}"
  --n-random "${N_RANDOM}"
  --num-threads "${SLURM_CPUS_PER_TASK}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${CENTRALS_ONLY:-0}" == "1" ]]; then
  CMD+=(--centrals-only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
