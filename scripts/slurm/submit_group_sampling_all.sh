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
#   SUBVOLS_L800=1,2,4,...,1024
#   SUBVOLS_MILL1=1,2,4,...,64
#   SUBVOLS_MILL2=1,2,4,...,64
#   CPUS_PER_TASK=32
#   MEM_PER_TASK=96G
#   TIME_LIMIT=6:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_group_sampling_grid.slurm"
MODEL_NAME="${MODEL_NAME:-gp14}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/convergence/group_sampling_jobs/${MODEL_NAME}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM_PER_TASK="${MEM_PER_TASK:-96G}"
TIME_LIMIT="${TIME_LIMIT:-6:00:00}"

SUBVOLS_L800="${SUBVOLS_L800:-1,2,4,8,10,15,20,25,30,50,100,200,400,600,800,1024}"
SUBVOLS_MILL1="${SUBVOLS_MILL1:-1,2,4,8,10,15,20,25,30,40,50,64}"
SUBVOLS_MILL2="${SUBVOLS_MILL2:-1,2,4,8,10,15,20,25,30,40,50,64}"

expand_subvols() {
    local spec="$1"
    local nmax="$2"

    if [[ -z "${spec}" ]]; then
        return 0
    fi

    if [[ "${spec}" == *-* && "${spec}" != *,* ]]; then
        local start end
        IFS='-' read -r start end <<<"${spec}"
        seq "${start}" "${end}" | awk -v nmax="${nmax}" '{n=$1+0; if (n>=1 && n<=nmax) print n}'
        return 0
    fi

    tr ',' '\n' <<<"${spec}" \
        | awk -v nmax="${nmax}" '{gsub(/^[ \t]+|[ \t]+$/, "", $0); if ($0 ~ /^[0-9]+$/) {n=$0+0; if (n>=1 && n<=nmax) print n}}'
}

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Missing SLURM template: ${TEMPLATE}"
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

SIMS_CSV="${SIMS_CSV:-L800,Mill1,Mill2}"
read -r -a SIMS <<<"${SIMS_CSV//,/ }"
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
            SUBVOLS_SPEC="${SUBVOLS_L800}"
            BOXSIZE=542.16
            ;;
        Mill1)
            NMAX=64
            SUBVOLS_SPEC="${SUBVOLS_MILL1}"
            BOXSIZE=365.0
            ;;
        Mill2)
            NMAX=64
            SUBVOLS_SPEC="${SUBVOLS_MILL2}"
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

    mapfile -t SUBVOL_LIST < <(expand_subvols "${SUBVOLS_SPEC}" "${NMAX}" | sort -n -u)
    if [[ ${#SUBVOL_LIST[@]} -eq 0 ]]; then
        echo "No valid n_subvol values for ${SIM} using SUBVOLS='${SUBVOLS_SPEC}'"
        continue
    fi

    for IZ in "${IZ_LIST[@]}"; do
        for MODE in "${MODES[@]}"; do
            for N_SUBVOL in "${SUBVOL_LIST[@]}"; do
                JOB_NAME="gsamp_${MODEL_NAME}_${MODE}_${SIM}_iz${IZ}_n${N_SUBVOL}"
                OUT_DIR="${OUT_ROOT}/${MODE}/${SIM}"

                echo "Submitting ${JOB_NAME} (n_subvol=${N_SUBVOL}, boxsize=${BOXSIZE}, cpus=${CPUS_PER_TASK}, mem=${MEM_PER_TASK})"
                sbatch \
                    --job-name="${JOB_NAME}" \
                    --time="${TIME_LIMIT}" \
                    --cpus-per-task="${CPUS_PER_TASK}" \
                    --mem="${MEM_PER_TASK}" \
                    --output="${REPO_ROOT}/logs/${JOB_NAME}_%j.log" \
                    --error="${REPO_ROOT}/logs/${JOB_NAME}_%j.err" \
                    --export=ALL,MODEL_NAME="${MODEL_NAME}",MODE="${MODE}",SIM_NAME="${SIM}",IZ="${IZ}",NMAX="${NMAX}",N_SUBVOL="${N_SUBVOL}",OUTPUT_DIR="${OUT_DIR}",BOXSIZE="${BOXSIZE}" \
                    "${TEMPLATE}"

                TOTAL=$((TOTAL + 1))
            done
        done
    done

done

echo "Submitted ${TOTAL} jobs."
