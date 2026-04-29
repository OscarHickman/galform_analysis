#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -J convergence_analysis
#SBATCH -o logs/convergence_%A_%a.log
#SBATCH -e logs/convergence_%A_%a.err
#SBATCH -p cosma5
#SBATCH -A durham
#SBATCH -t 10:00:00
#SBATCH --array=0-1

# Set workspace
cd /cosma/apps/durham/dc-hick2/galform_analysis

# Create directories
mkdir -p logs data/convergence/convergence_results

# Load Python module
module load python/3.9.19

# Export PYTHONPATH to include site-packages from venv
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

# Define redshifts and subvolume counts
redshifts=(271 207)
subvols="1,2,4,8,20,50,100,300,600,1000"
output_dir="data/convergence/convergence_results"

# Get the redshift for this array job
iz=${redshifts[$SLURM_ARRAY_TASK_ID]}

# Run the convergence computation
echo "Starting convergence computation for iz${iz} on $(date)"

python scripts/compute_convergence_specific.py \
    --iz "${iz}" \
    --subvols "${subvols}" \
    --output-dir "${output_dir}"

if [ $? -eq 0 ]; then
    echo "✓ Completed iz${iz} on $(date)"
else
    echo "✗ Failed iz${iz} on $(date)"
    exit 1
fi
