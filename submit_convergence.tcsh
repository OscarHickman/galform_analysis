#!/bin/tcsh -ef
#
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -J convergence_analysis
#SBATCH -o logs/convergence_%j.log
#SBATCH -p cosma5
#SBATCH -A durham
#SBATCH -t 10:00:00
#SBATCH --array=0-1
#

# Load environment
source /cosma/home/durham/dc-hick2/galform_analysis/.venv/bin/activate.csh

# Set output directory
set output_dir = "convergence_results"
mkdir -p $output_dir
mkdir -p logs

# Define redshifts and parameters
set redshifts = (271 207)
set max_ivols = "all"

# Get the redshift for this job
set idx = ${SLURM_ARRAY_TASK_ID}
set iz = $redshifts[$((${idx} + 1))]

# Run the convergence computation
echo "Starting convergence computation for iz${iz} on $(date)"
cd /cosma/home/durham/dc-hick2/galform_analysis

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
