#!/bin/bash
# Submit separate SLURM jobs per (simulation, redshift, mode) for notebook-style
# correlation convergence.
#
# Defaults:
# - L800 uses subvolumes 1..1024
# - Mill1 uses subvolumes 1..64
# - Mill2 uses subvolumes 1..64
#
# Environment overrides:
#   MODEL_NAME=gp14
#   OUT_ROOT=/path/to/output
#   SUBVOLS_L800=1-1024
#   SUBVOLS_MILL1=1-64
#   SUBVOLS_MILL2=1-64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_group_sampling_grid.slurm"
MODEL_NAME="${MODEL_NAME:-gp14}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/convergence/group_sampling_jobs/${MODEL_NAME}}"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Missing SLURM template: ${TEMPLATE}"
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

SIMS=(L800 Mill1 Mill2)
TOTAL=0
MODES=(normal weighted)

for SIM in "${SIMS[@]}"; do
    BASE_DIR="/cosma5/data/durham/dc-hick2/Galform_Out/${SIM}/${MODEL_NAME}"
    if [[ ! -d "${BASE_DIR}" ]]; then
        echo "Skipping ${SIM}: missing ${BASE_DIR}"
        continue
    fi

    case "${SIM}" in
        L800)
            NMAX=1024
            SUBVOLS="${SUBVOLS_L800:-1-1024}"
            BOXSIZE=542.16
            ;;
        Mill1)
            NMAX=64
            SUBVOLS="${SUBVOLS_MILL1:-1-64}"
            BOXSIZE=365.0
            ;;
        Mill2)
            NMAX=64
            SUBVOLS="${SUBVOLS_MILL2:-1-64}"
            BOXSIZE=73.0
            ;;
        *)
            echo "Unknown simulation ${SIM}; skipping"
            continue
            ;;
    esac

    mapfile -t IZ_LIST < <(find "${BASE_DIR}" -maxdepth 1 -type d -name 'iz*' -printf '%f\n' | sed 's/^iz//' | sort -n)
    if [[ ${#IZ_LIST[@]} -eq 0 ]]; then
        echo "No iz directories found for ${SIM}"
        continue
    fi

    for IZ in "${IZ_LIST[@]}"; do
        for MODE in "${MODES[@]}"; do
            JOB_NAME="gsamp_${MODEL_NAME}_${MODE}_${SIM}_iz${IZ}"
            OUT_DIR="${OUT_ROOT}/${MODE}/${SIM}"

            echo "Submitting ${JOB_NAME} (subvols=${SUBVOLS}, boxsize=${BOXSIZE})"
            sbatch \
                --job-name="${JOB_NAME}" \
                --export=ALL,MODEL_NAME="${MODEL_NAME}",MODE="${MODE}",SIM_NAME="${SIM}",IZ="${IZ}",NMAX="${NMAX}",SUBVOLS="${SUBVOLS}",OUTPUT_DIR="${OUT_DIR}",BOXSIZE="${BOXSIZE}" \
                "${TEMPLATE}"

            TOTAL=$((TOTAL + 1))
        done
    done

done

echo "Submitted ${TOTAL} jobs."
