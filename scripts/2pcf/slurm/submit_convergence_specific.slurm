#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -J convergence_specific
#SBATCH -o logs/convergence_specific_%j.log
#SBATCH -e logs/convergence_specific_%j.err
#SBATCH -p cosma5
#SBATCH -A durham
#SBATCH -t 3-00:00:00

# Set workspace
cd /cosma/home/durham/dc-hick2/galform_analysis

# Create directories
mkdir -p logs data/convergence/convergence_results

# Load Python module
module load python/3.9.19

# Export PYTHONPATH to include site-packages from venv
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

if [ -z "$IZ" ] || [ -z "$MODE" ]; then
    echo "ERROR: IZ and MODE must be set (e.g., --export=IZ=271,MODE=corr)"
    exit 1
fi

subvols="1,2,4,8,20,60,100,300,600,1000,1024"
output_dir=${OUTPUT_DIR:-data/convergence/convergence_results}
mhalo_min=${MHALO_MIN:-1e9}
corr_include_satellites=${CORR_INCLUDE_SAT:-0}

# Run the convergence computation

echo "Starting convergence computation for iz${IZ} (${MODE}) on $(date)"

if [ "$corr_include_satellites" -eq 1 ] && [ "$MODE" = "corr" ]; then
    corr_sat_flag="--corr-include-satellites"
else
    corr_sat_flag=""
fi

python scripts/compute_convergence_specific_grouped.py \
    --iz "$IZ" \
    --subvols "$subvols" \
    --output-dir "$output_dir" \
    --mhalo-min "$mhalo_min" \
    --mode "$MODE" \
    $corr_sat_flag

if [ $? -eq 0 ]; then
    echo "Completed iz${IZ} (${MODE}) on $(date)"
else
    echo "Failed iz${IZ} (${MODE}) on $(date)"
    exit 1
fi
