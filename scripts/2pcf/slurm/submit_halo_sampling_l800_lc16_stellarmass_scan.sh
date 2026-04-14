#!/bin/bash
# Submit halo-sampling-corrected 2PCF jobs for L800/lc16 while varying stellar-mass cut.
# Fixed settings per request:
#   - centrals_only = 0 (include centrals + satellites)
#   - mhalo_min     = 1e9
#   - n_total_subvolumes = 1024

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_halo_sampling_grid.slurm"
PARTITION="cosma5"

# Slurm array indices are typically limited to [1, MaxArraySize-1].
MAX_ARRAY_TASK_ID="${MAX_ARRAY_TASK_ID:-}"
if [[ -z "${MAX_ARRAY_TASK_ID}" ]]; then
    max_array_size_raw="$( (scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/{gsub(/ /, "", $2); print $2; exit}') || true )"
    if [[ "${max_array_size_raw}" =~ ^[0-9]+$ ]] && (( max_array_size_raw > 1 )); then
        MAX_ARRAY_TASK_ID=$((max_array_size_raw - 1))
    else
        MAX_ARRAY_TASK_ID=1000
    fi
fi

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/halo_sampling_mstar_scan}"
MODE="${MODE:-halo}"
SIM_NAME="L800"
MODEL_NAME="lc16"
BOXSIZE="542.16"

IZ_LIST=(271 207 155)
NMAX="1024"
N_SUBVOL_LIST="1,2,4,8,10,15,20,25,30,50,100,200,400,600,800,1024"

# "none" means no stellar-mass cut is passed to the compute script.
MSTAR_LIST=(none 8.5 9.0 9.5 10.0)
MHALO_MIN="1e9"
CENTRALS_ONLY="0"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Missing SLURM template: ${TEMPLATE}"
    exit 1
fi

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Base directory not found: ${BASE_DIR}"
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

queued_names=""
if command -v squeue >/dev/null 2>&1; then
    queued_names="$(squeue -u "${USER}" -h -o "%j" || true)"
fi

submitted=0
skipped_existing=0
skipped_queued=0

all_outputs_exist_for_spec() {
    local out_dir_base="$1"
    local iz="$2"
    local spec="$3"
    local out_csv

    IFS=',' read -r -a __nvals <<<"${spec}"
    local n
    for n in "${__nvals[@]}"; do
        out_csv="${out_dir_base}/nsubvol_${n}/halo_sampling_convergence_${MODE}_${SIM_NAME}_iz${iz}.csv"
        if [[ ! -f "${out_csv}" ]]; then
            return 1
        fi
    done
    return 0
}

is_queued() {
    local name="$1"
    [[ -n "${queued_names}" ]] && grep -qx "${name}" <<<"${queued_names}"
}

submit_array_job() {
    local job_name="$1"
    local array_spec="$2"
    local iz="$3"
    local out_dir_base="$4"
    local mstar="$5"

    if is_queued "${job_name}"; then
        echo "Skipping (already queued): ${job_name}"
        skipped_queued=$((skipped_queued + 1))
        return
    fi

    if all_outputs_exist_for_spec "${out_dir_base}" "${iz}" "${array_spec}"; then
        echo "Skipping (all per-n outputs exist): ${out_dir_base} [${array_spec}]"
        skipped_existing=$((skipped_existing + 1))
        return
    fi

    mkdir -p "${out_dir_base}"
    echo "Submitting ${job_name} as array ${array_spec}"

    local export_args="ALL,MODE=${MODE},SIM_NAME=${SIM_NAME},MODEL_NAME=${MODEL_NAME},IZ=${iz},NMAX=${NMAX},OUTPUT_DIR_BASE=${out_dir_base},BOXSIZE=${BOXSIZE},MHALO_MIN=${MHALO_MIN},CENTRALS_ONLY=${CENTRALS_ONLY},BASE_DIR_OVERRIDE=${BASE_DIR}"
    if [[ "${mstar}" != "none" ]]; then
        export_args+=",MSTAR_MIN_LOG10=${mstar}"
    fi

    sbatch \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --array="${array_spec}" \
        --export="${export_args}" \
        "${TEMPLATE}"

    submitted=$((submitted + 1))
}

submit_single_job() {
    local job_name="$1"
    local n="$2"
    local iz="$3"
    local out_dir_base="$4"
    local mstar="$5"

    local out_dir="${out_dir_base}/nsubvol_${n}"
    local out_csv="${out_dir}/halo_sampling_convergence_${MODE}_${SIM_NAME}_iz${iz}.csv"

    if [[ -f "${out_csv}" ]]; then
        echo "Skipping (output exists): ${out_csv}"
        skipped_existing=$((skipped_existing + 1))
        return
    fi

    if is_queued "${job_name}"; then
        echo "Skipping (already queued): ${job_name}"
        skipped_queued=$((skipped_queued + 1))
        return
    fi

    mkdir -p "${out_dir}"
    echo "Submitting ${job_name} (n_subvol=${n})"

    local export_args="ALL,MODE=${MODE},SIM_NAME=${SIM_NAME},MODEL_NAME=${MODEL_NAME},IZ=${iz},NMAX=${NMAX},SUBVOLS=${n},OUTPUT_DIR=${out_dir},BOXSIZE=${BOXSIZE},MHALO_MIN=${MHALO_MIN},CENTRALS_ONLY=${CENTRALS_ONLY},BASE_DIR_OVERRIDE=${BASE_DIR}"
    if [[ "${mstar}" != "none" ]]; then
        export_args+=",MSTAR_MIN_LOG10=${mstar}"
    fi

    sbatch \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --export="${export_args}" \
        "${TEMPLATE}"

    submitted=$((submitted + 1))
}

for iz in "${IZ_LIST[@]}"; do
    for mstar in "${MSTAR_LIST[@]}"; do
        if [[ "${mstar}" == "none" ]]; then
            mstar_tag="mstar_none"
            mstar_job="none"
        else
            mstar_tag="mstar_${mstar}"
            mstar_job="m${mstar//./p}"
        fi

        out_dir_base="${OUT_ROOT}/${MODEL_NAME}/iz${iz}/ntotal_${NMAX}/custom/mhalo_1e9/centrals_0/${mstar_tag}"
        job_name="hsL8lc16_i${iz}_N${NMAX}_m1e9_c0_${mstar_job}_alist"
        IFS=',' read -r -a __nvals <<<"${N_SUBVOL_LIST}"
        csv_le=""
        for n in "${__nvals[@]}"; do
            if (( n <= MAX_ARRAY_TASK_ID )); then
                if [[ -z "${csv_le}" ]]; then
                    csv_le="${n}"
                else
                    csv_le="${csv_le},${n}"
                fi
            else
                submit_single_job "${job_name}_n${n}" "${n}" "${iz}" "${out_dir_base}" "${mstar}"
            fi
        done

        if [[ -n "${csv_le}" ]]; then
            submit_array_job "${job_name}" "${csv_le}" "${iz}" "${out_dir_base}" "${mstar}"
        fi
    done
done

echo "------------------------------------------------------------"
echo "Stellar-mass scan submission summary"
echo "  submitted:        ${submitted}"
echo "  skipped existing: ${skipped_existing}"
echo "  skipped queued:   ${skipped_queued}"
echo "------------------------------------------------------------"
