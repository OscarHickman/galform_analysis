#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_rsd_subvol_grid.slurm"

SIM_NAME="${SIM_NAME:-L800}"
MODEL_NAME="${MODEL_NAME:-lc16}"
IZ="${IZ:-155}"
NMAX="${NMAX:-1024}"
BOXSIZE="${BOXSIZE:-542.16}"
MHALO_MIN="${MHALO_MIN:-1e10}"
N_RANDOM="${N_RANDOM:-120000}"
CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
SEED="${SEED:-42}"
IVOL_SELECTION="${IVOL_SELECTION:-first}"

SUBVOLS_CSV="${SUBVOLS_CSV:-10,20,50,100,400,800,1024}"
MODES_CSV="${MODES_CSV:-normal,corrected}"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/${SIM_NAME}/${MODEL_NAME}}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${REPO_ROOT}/data/redshift_space_distortions/subvol_jobs/${MODEL_NAME}/${SIM_NAME}/iz${IZ}}"

IFS=',' read -r -a SUBVOLS <<< "${SUBVOLS_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"

total=0
for mode in "${MODES[@]}"; do
  for n in "${SUBVOLS[@]}"; do
    JOB_NAME="rsd_${MODEL_NAME}_${SIM_NAME}_iz${IZ}_${mode}_n${n}_${IVOL_SELECTION}"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${mode}"
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting ${JOB_NAME}"
    sbatch \
      --job-name="${JOB_NAME}" \
      --partition=cosma5 \
      --export=ALL,SIM_NAME="${SIM_NAME}",MODEL_NAME="${MODEL_NAME}",IZ="${IZ}",NMAX="${NMAX}",N_SUBVOL="${n}",MODE="${mode}",IVOL_SELECTION="${IVOL_SELECTION}",SEED="${SEED}",BOXSIZE="${BOXSIZE}",MHALO_MIN="${MHALO_MIN}",N_RANDOM="${N_RANDOM}",CENTRALS_ONLY="${CENTRALS_ONLY}",BASE_DIR="${BASE_DIR}",OUTPUT_DIR="${OUTPUT_DIR}" \
      "${TEMPLATE}"
    total=$((total + 1))
  done
done

echo "Submitted ${total} RSD subvolume jobs."
