#!/bin/tcsh -ef
#
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
cd /cosma/home/durham/dc-hick2/galform_analysis

# Create directories first (before SLURM tries to write logs)
mkdir -p logs convergence_results

# Load environment
source .venv/bin/activate.csh

# Define redshifts
set redshifts = (271 207)
set max_ivols = "all"
set output_dir = "convergence_results"

# Get the redshift for this array job
set idx = ${SLURM_ARRAY_TASK_ID}
set iz = $redshifts[$((${idx} + 1))]

# Run the convergence computation
echo "Starting convergence computation for iz${iz} on $(date)"

python compute_convergence.py \
    --iz ${iz} \
    --max-ivols ${max_ivols} \
    --output-dir ${output_dir}

if ($status == 0) then
    echo "✓ Completed iz${iz} on $(date)"
else
    echo "✗ Failed iz${iz} on $(date)"
    exit 1
endif
