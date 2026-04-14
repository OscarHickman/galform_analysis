#!/bin/bash
# Submit full convergence matrices for requested selection sets.
#
# Default sets:
#  - mhalo1e9_all,  mhalo1e9_cen
#  - mhalo1e10_all, mhalo1e10_cen
#  - mhalo1e11_all, mhalo1e11_cen
#
# For each set, submit both modes: normal and weighted.
# Subvolume lists:
#  - L800: 1,2,4,...,1024
#  - Mill1: 1,2,4,...,64
#  - Mill2: 1,2,4,...,64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_group_sampling_grid.slurm"
MODEL_NAME="${MODEL_NAME:-gp14}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/convergence/group_sampling_halo_filter_sets/${MODEL_NAME}}"
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
MODES=(normal weighted)
# Comma-separated set list can be overridden, e.g.
#   SETS_CSV="mhalo1e9_all,mhalo1e9_cen"
SETS_CSV="${SETS_CSV:-mhalo1e9_all,mhalo1e9_cen,mhalo1e10_all,mhalo1e10_cen,mhalo1e11_all,mhalo1e11_cen}"
read -r -a SETS <<<"${SETS_CSV//,/ }"
TOTAL=0

for SET_NAME in "${SETS[@]}"; do
    case "${SET_NAME}" in
        mhalo1e9_all)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e9
            ;;
        mhalo1e9_cen)
            CENTRALS_ONLY=1
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e9
            ;;
        mhalo1e10_all)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e10
            ;;
        mhalo1e10_cen)
            CENTRALS_ONLY=1
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e10
            ;;
        mhalo1e11_all)
            CENTRALS_ONLY=0
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e11
            ;;
        mhalo1e11_cen)
            CENTRALS_ONLY=1
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=1e11
            ;;
        centrals_only)
            CENTRALS_ONLY=1
            MSTAR_MIN_LOG10=-99.0
            MHALO_MIN=0
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
                    JOB_NAME="gsamp_${MODEL_NAME}_${SET_NAME}_${MODE}_${SIM}_iz${IZ}_n${N_SUBVOL}"
                    OUT_DIR="${OUT_ROOT}/${SET_NAME}/${MODE}/${SIM}"

                    echo "Submitting ${JOB_NAME} (n_subvol=${N_SUBVOL}, boxsize=${BOXSIZE}, mhalo_min=${MHALO_MIN}, centrals_only=${CENTRALS_ONLY}, cpus=${CPUS_PER_TASK}, mem=${MEM_PER_TASK})"
                    sbatch \
                        --job-name="${JOB_NAME}" \
                        --time="${TIME_LIMIT}" \
                        --cpus-per-task="${CPUS_PER_TASK}" \
                        --mem="${MEM_PER_TASK}" \
                        --output="${REPO_ROOT}/logs/${JOB_NAME}_%j.log" \
                        --error="${REPO_ROOT}/logs/${JOB_NAME}_%j.err" \
                        --export=ALL,MODEL_NAME="${MODEL_NAME}",MODE="${MODE}",SIM_NAME="${SIM}",IZ="${IZ}",NMAX="${NMAX}",N_SUBVOL="${N_SUBVOL}",OUTPUT_DIR="${OUT_DIR}",BOXSIZE="${BOXSIZE}",MSTAR_MIN_LOG10="${MSTAR_MIN_LOG10}",CENTRALS_ONLY="${CENTRALS_ONLY}",MHALO_MIN="${MHALO_MIN}" \
                        "${TEMPLATE}"

                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done

echo "Submitted ${TOTAL} jobs in total."
