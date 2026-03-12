#!/bin/bash
#SBATCH --job-name=pair_counts
#SBATCH --output=logs/pair_counts_%j.out
#SBATCH --error=logs/pair_counts_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=cosma8-shm
#SBATCH --account=durham

# Load Python module
module load python/3.9.19

# Export PYTHONPATH to include site-packages from venv
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script
cd /cosma/home/durham/dc-hick2/galform_analysis

echo "Starting pair counts computation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "=================================="

OUTPUT_DIR=${OUTPUT_DIR:-data/convergence/convergence_results}

if [ -n "$IZ" ]; then
	echo "Running pair counts for iz${IZ}"
	python scripts/compute_pair_counts.py --output-dir "$OUTPUT_DIR" --iz "$IZ"
else
	python scripts/compute_pair_counts.py --output-dir "$OUTPUT_DIR"
fi

echo "=================================="
echo "Pair counts computation finished!"
echo "Time: $(date)"
