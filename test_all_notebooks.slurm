#!/bin/bash
#SBATCH --job-name=test_all_nb
#SBATCH --partition=cosma5
#SBATCH --account=durham
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/test_all_nb_%j.log

module load python/3.9.19
source .venv/bin/activate
pytest --nbmake examples/
