#!/bin/bash
# Submit all 12 weighted convergence jobs:
#   3 redshifts (iz207, iz271, iz176) x 2 galaxy types (cen, all) x 2 mass cuts (1e9, 1e10)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/submit_weighted_convergence_run.sh"
BASE_OUTDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/data/convergence/convergence_results_weighted"

IZ_LIST=(207 271 176)

for IZ in "${IZ_LIST[@]}"; do
    for MASS_TAG in mhalo1e9 mhalo1e10; do
        if [[ "${MASS_TAG}" == "mhalo1e9" ]]; then
            MHALO_MIN=1e9
        else
            MHALO_MIN=1e10
        fi

        for GAL_TAG in cen all; do
            if [[ "${GAL_TAG}" == "cen" ]]; then
                INCLUDE_SATELLITES=0
            else
                INCLUDE_SATELLITES=1
            fi

            OUTPUT_DIR="${BASE_OUTDIR}/${MASS_TAG}_${GAL_TAG}"
            JOB_NAME="wconv_iz${IZ}_${MASS_TAG}_${GAL_TAG}"

            echo "Submitting: iz=${IZ} mass=${MHALO_MIN} gal=${GAL_TAG} -> ${OUTPUT_DIR}"

            sbatch \
                --job-name="${JOB_NAME}" \
                --export=ALL,IZ_NUM=${IZ},MHALO_MIN=${MHALO_MIN},INCLUDE_SATELLITES=${INCLUDE_SATELLITES},GAL_TAG=${GAL_TAG},MASS_TAG=${MASS_TAG},OUTPUT_DIR="${OUTPUT_DIR}" \
                "${TEMPLATE}"
        done
    done
done

echo "All 12 jobs submitted."
