#!/bin/bash
# Submit Corrfunc full-box reference jobs for L800/lc16 with only n_subvol=1024.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/run_corrfunc_fullbox_reference.slurm"
PARTITION="cosma5"

BASE_DIR="${BASE_DIR:-/cosma5/data/durham/dc-hick2/Galform_Out/L800/lc16}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/halo_sampling_5_corrfunc_fullbox_1024}"
SIM_NAME="L800"
MODEL_NAME="lc16"
N_SUBVOL="1024"
IVOL_START="0"

# Match the same matrix dimensions as previous campaigns, but only for N=1024.
IZ_LIST=(271 207 155)
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

is_queued() {
    local name="$1"
    [[ -n "${queued_names}" ]] && grep -qx "${name}" <<<"${queued_names}"
}

for iz in "${IZ_LIST[@]}"; do
    for mhalo in "${MHALO_LIST[@]}"; do
        # Keep scientific notation tags like 1e13 while sanitizing unusual characters.
        mhalo_tag="${mhalo//[^0-9a-zA-Z._-]/_}"

        for centrals in "${CENTRALS_LIST[@]}"; do
            out_dir="${OUT_ROOT}/${MODEL_NAME}/iz${iz}/ntotal_1024/custom/mhalo_${mhalo_tag}/centrals_${centrals}/nsubvol_1024"
            out_csv="${out_dir}/halo_sampling_convergence_corrfunc_${SIM_NAME}_iz${iz}.csv"
            job_name="cfL8lc16_i${iz}_N1024_${mhalo_tag}_c${centrals}"

            if [[ -f "${out_csv}" ]]; then
                echo "Skipping (output exists): ${out_csv}"
                skipped_existing=$((skipped_existing + 1))
                continue
            fi

            if is_queued "${job_name}"; then
                echo "Skipping (already queued): ${job_name}"
                skipped_queued=$((skipped_queued + 1))
                continue
            fi

            mkdir -p "${out_dir}"
            echo "Submitting ${job_name}"
            sbatch \
                --partition="${PARTITION}" \
                --job-name="${job_name}" \
                --export=ALL,SIM_NAME="${SIM_NAME}",MODEL_NAME="${MODEL_NAME}",IZ="${iz}",N_SUBVOL="${N_SUBVOL}",IVOL_START="${IVOL_START}",OUTPUT_DIR="${out_dir}",MHALO_MIN="${mhalo}",CENTRALS_ONLY="${centrals}",BASE_DIR_OVERRIDE="${BASE_DIR}" \
                "${TEMPLATE}"

            submitted=$((submitted + 1))
        done
    done
done

echo "------------------------------------------------------------"
echo "Submission summary"
echo "  submitted:        ${submitted}"
echo "  skipped existing: ${skipped_existing}"
echo "  skipped queued:   ${skipped_queued}"
echo "------------------------------------------------------------"
