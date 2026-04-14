#!/bin/bash
# Submit a single COSMA5 full-box standard RSD multipole job (all 1024 subvolumes).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_rsd_fullbox.slurm"

SIM_NAME="${SIM_NAME:-L800}"
MODEL_NAME="${MODEL_NAME:-lc16}"
IZ="${IZ:-155}"
NMAX="${NMAX:-1024}"
BOXSIZE="${BOXSIZE:-542.16}"
MHALO_MIN="${MHALO_MIN:-1e10}"
N_RANDOM="${N_RANDOM:-200000}"
CENTRALS_ONLY="${CENTRALS_ONLY:-0}"
MAX_GALAXIES_PER_IVOL="${MAX_GALAXIES_PER_IVOL:-0}"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/${SIM_NAME}/${MODEL_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/redshift_space_distortions/fullbox/${MODEL_NAME}/${SIM_NAME}}"

if [[ ! -f "${TEMPLATE}" ]]; then
	echo "Missing SLURM template: ${TEMPLATE}"
	exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"

JOB_NAME="rsd_${MODEL_NAME}_${SIM_NAME}_iz${IZ}_fullbox"
echo "Submitting ${JOB_NAME} on cosma5"

sbatch \
	--job-name="${JOB_NAME}" \
	--export=ALL,SIM_NAME="${SIM_NAME}",MODEL_NAME="${MODEL_NAME}",IZ="${IZ}",NMAX="${NMAX}",BOXSIZE="${BOXSIZE}",MHALO_MIN="${MHALO_MIN}",N_RANDOM="${N_RANDOM}",CENTRALS_ONLY="${CENTRALS_ONLY}",MAX_GALAXIES_PER_IVOL="${MAX_GALAXIES_PER_IVOL}",BASE_DIR="${BASE_DIR}",OUTPUT_DIR="${OUTPUT_DIR}" \
	"${TEMPLATE}"
