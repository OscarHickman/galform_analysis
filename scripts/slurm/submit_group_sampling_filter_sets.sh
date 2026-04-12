#!/bin/bash
# Submit full convergence matrices for requested selection sets:
#  - centrals_only
#  - mhalo >= 1e9
#  - mhalo >= 1e10
#  - mhalo >= 1e11
#
# For each set, submit both modes: normal and weighted.
# Subvolume ranges:
#  - L800: 1-1024
#  - Mill1: 1-64
#  - Mill2: 1-64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_group_sampling_grid.slurm"
MODEL_NAME="${MODEL_NAME:-gp14}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/convergence/group_sampling_halo_filter_sets/${MODEL_NAME}}"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Missing SLURM template: ${TEMPLATE}"
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

SIMS=(L800 Mill1 Mill2)
MODES=(normal weighted)
SETS=(centrals_only mhalo1e9 mhalo1e10 mhalo1e11)
TOTAL=0

for SET_NAME in "${SETS[@]}"; do
    case "${SET_NAME}" in
        centrals_only)
            CENTRALS_ONLY=1
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=0
            ;;
        mhalo1e9)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e9
            ;;
        mhalo1e10)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e10
            ;;
        mhalo1e11)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e11
            ;;
        *)
            echo "Unknown set ${SET_NAME}; skipping"
            continue
            ;;
    esac

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
        esac

        mapfile -t IZ_LIST < <(find "${BASE_DIR}" -maxdepth 1 -type d -name 'iz*' -printf '%f\n' | sed 's/^iz//' | sort -n)
        if [[ ${#IZ_LIST[@]} -eq 0 ]]; then
            echo "No iz directories found for ${SIM}"
            continue
        fi

        for IZ in "${IZ_LIST[@]}"; do
            for MODE in "${MODES[@]}"; do
                JOB_NAME="gsamp_${MODEL_NAME}_${SET_NAME}_${MODE}_${SIM}_iz${IZ}"
                OUT_DIR="${OUT_ROOT}/${SET_NAME}/${MODE}/${SIM}"

                echo "Submitting ${JOB_NAME} (subvols=${SUBVOLS}, boxsize=${BOXSIZE}, mhalo_min=${MHALO_MIN}, centrals_only=${CENTRALS_ONLY}, mstar_min_log10=${MSTAR_MIN_LOG10})"
                sbatch \
                    --job-name="${JOB_NAME}" \
                    --export=ALL,MODEL_NAME="${MODEL_NAME}",MODE="${MODE}",SIM_NAME="${SIM}",IZ="${IZ}",NMAX="${NMAX}",SUBVOLS="${SUBVOLS}",OUTPUT_DIR="${OUT_DIR}",BOXSIZE="${BOXSIZE}",MSTAR_MIN_LOG10="${MSTAR_MIN_LOG10}",CENTRALS_ONLY="${CENTRALS_ONLY}",MHALO_MIN="${MHALO_MIN}" \
                    "${TEMPLATE}"

                TOTAL=$((TOTAL + 1))
            done
        done
    done
done

echo "Submitted ${TOTAL} jobs in total."
