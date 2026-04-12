#!/bin/bash
# Submit random-subvolume weighted-estimator 2PCF jobs for L800/lc16 over
# requested redshifts, halo-mass cuts, centrals-only settings, and n_subvol values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_subvol_weighted_grid_random.slurm"
PARTITION="cosma5"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/halo_sampling_4_subvol_weighted_random}"
MODE="${MODE:-weighted}"
PARTITION_SCHEME="${PARTITION_SCHEME:-ivol}"
RANDOM_SELECTION_SEED="${RANDOM_SELECTION_SEED:-314159}"
SIM_NAME="L800"
MODEL_NAME="lc16"
BOXSIZE="542.16"
K_TOTAL="1024"

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

# Requested redshifts and matrix definition.
IZ_LIST=(271 207 155)
N_SUBVOL_LIST="1,2,4,8,10,15,20,25,30,50,100,200,400,600,800,1024"
# Comma-separated mass-cut list can be overridden, e.g. MHALO_LIST_CSV="1e13"
MHALO_LIST_CSV="${MHALO_LIST_CSV:-1e11,1e9}"
read -r -a MHALO_LIST <<<"${MHALO_LIST_CSV//,/ }"
CENTRALS_LIST=(0 1)

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

    if [[ "${spec}" == *-* && "${spec}" != *,* ]]; then
        local start end n
        start="${spec%-*}"
        end="${spec#*-}"
        for n in $(seq "${start}" "${end}"); do
            out_csv="${out_dir_base}/nsubvol_${n}/halo_sampling_convergence_${MODE}_${SIM_NAME}_iz${iz}.csv"
            if [[ ! -f "${out_csv}" ]]; then
                return 1
            fi
        done
        return 0
    fi

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
    local nmax="$4"
    local mhalo="$5"
    local centrals="$6"
    local out_dir_base="$7"

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
    sbatch \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --array="${array_spec}" \
        --export=ALL,MODE="${MODE}",PARTITION_SCHEME="${PARTITION_SCHEME}",RANDOM_SELECTION_SEED="${RANDOM_SELECTION_SEED}",SIM_NAME="${SIM_NAME}",MODEL_NAME="${MODEL_NAME}",IZ="${iz}",NMAX="${nmax}",K_TOTAL="${K_TOTAL}",OUTPUT_DIR_BASE="${out_dir_base}",BOXSIZE="${BOXSIZE}",MHALO_MIN="${mhalo}",CENTRALS_ONLY="${centrals}",BASE_DIR_OVERRIDE="${BASE_DIR}" \
        "${TEMPLATE}"

    submitted=$((submitted + 1))
}

submit_single_job() {
    local job_name="$1"
    local n="$2"
    local iz="$3"
    local nmax="$4"
    local mhalo="$5"
    local centrals="$6"
    local out_dir_base="$7"

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
    sbatch \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --export=ALL,MODE="${MODE}",PARTITION_SCHEME="${PARTITION_SCHEME}",RANDOM_SELECTION_SEED="${RANDOM_SELECTION_SEED}",SIM_NAME="${SIM_NAME}",MODEL_NAME="${MODEL_NAME}",IZ="${iz}",NMAX="${nmax}",K_TOTAL="${K_TOTAL}",SUBVOLS="${n}",OUTPUT_DIR="${out_dir}",BOXSIZE="${BOXSIZE}",MHALO_MIN="${mhalo}",CENTRALS_ONLY="${centrals}",BASE_DIR_OVERRIDE="${BASE_DIR}" \
        "${TEMPLATE}"

    submitted=$((submitted + 1))
}

submit_spec() {
    local iz="$1"
    local nmax="$2"
    local spec="$3"
    local set_tag="$4"
    local mhalo="$5"
    local mhalo_tag="$6"
    local centrals="$7"

    local out_dir_base="${OUT_ROOT}/${MODEL_NAME}/iz${iz}/ntotal_${nmax}/${set_tag}/mhalo_${mhalo_tag}/centrals_${centrals}"

    local base_job="swrL8lc16_i${iz}_N${nmax}_${set_tag}_${mhalo_tag}_c${centrals}"

    if [[ "${spec}" == *-* && "${spec}" != *,* ]]; then
        local start end
        start="${spec%-*}"
        end="${spec#*-}"

        if (( end <= MAX_ARRAY_TASK_ID )); then
            submit_array_job "${base_job}_a${start}_${end}" "${spec}" "${iz}" "${nmax}" "${mhalo}" "${centrals}" "${out_dir_base}"
            return
        fi

        if (( start <= MAX_ARRAY_TASK_ID )); then
            submit_array_job "${base_job}_a${start}_${MAX_ARRAY_TASK_ID}" "${start}-${MAX_ARRAY_TASK_ID}" "${iz}" "${nmax}" "${mhalo}" "${centrals}" "${out_dir_base}"
        fi

        local n
        for n in $(seq $((MAX_ARRAY_TASK_ID + 1)) "${end}"); do
            submit_single_job "${base_job}_n${n}" "${n}" "${iz}" "${nmax}" "${mhalo}" "${centrals}" "${out_dir_base}"
        done
        return
    fi

    IFS=',' read -r -a __nvals <<<"${spec}"
    local n
    local csv_le=""
    for n in "${__nvals[@]}"; do
        if (( n <= MAX_ARRAY_TASK_ID )); then
            if [[ -z "${csv_le}" ]]; then
                csv_le="${n}"
            else
                csv_le="${csv_le},${n}"
            fi
        else
            submit_single_job "${base_job}_n${n}" "${n}" "${iz}" "${nmax}" "${mhalo}" "${centrals}" "${out_dir_base}"
        fi
    done

    if [[ -n "${csv_le}" ]]; then
        submit_array_job "${base_job}_alist" "${csv_le}" "${iz}" "${nmax}" "${mhalo}" "${centrals}" "${out_dir_base}"
    fi
}

for iz in "${IZ_LIST[@]}"; do
    for mhalo in "${MHALO_LIST[@]}"; do
        # Keep human-readable scientific notation tags like 1e13 while
        # sanitizing any uncommon characters for paths/job names.
        mhalo_tag="${mhalo//[^0-9a-zA-Z._-]/_}"
        for centrals in "${CENTRALS_LIST[@]}"; do
            submit_spec "${iz}" "1024" "${N_SUBVOL_LIST}" "custom" "${mhalo}" "${mhalo_tag}" "${centrals}"
        done
    done
done

echo "------------------------------------------------------------"
echo "Submission summary"
echo "  submitted:        ${submitted}"
echo "  skipped existing: ${skipped_existing}"
echo "  skipped queued:   ${skipped_queued}"
echo "------------------------------------------------------------"