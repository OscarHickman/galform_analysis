#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -J conv_iter_v3
#SBATCH -o logs/convergence_v3_iz%a_iter%A.log
#SBATCH -e logs/convergence_v3_iz%a_iter%A.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 12:00:00
#SBATCH --array=0-5

# Convergence analysis v3 (CORRECTED overlapping subvolumes) with 3 iterations for iz271 and iz207
# Subvolumes: 1,2,3,4,5,8,10,15,20,25,30,40,50,80,100,150,200,300,500,750,1024
# CRITICAL FIXES:
#   - HMF: Combines all halos into SAME volume (not sum volumes)
#   - 2PCF: Combines all galaxies into SAME box (not average xi)
#   - Single subvolume already gives unbiased estimate (1/1024 factors cancel)
#   - More subvolumes = better statistics (reduced shot noise)

cd /cosma/home/durham/dc-hick2/galform_analysis

mkdir -p logs convergence_results_v3_iter1 convergence_results_v3_iter2 convergence_results_v3_iter3

module load python/3.9.19
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

subvols="1,2,3,4,5,8,10,15,20,25,30,40,50,80,100,150,200,300,500,750,1024"

if [ $SLURM_ARRAY_TASK_ID -le 2 ]; then
    iz=271
    iter=$((SLURM_ARRAY_TASK_ID + 1))
else
    iz=207
    iter=$((SLURM_ARRAY_TASK_ID - 2))
fi

output_dir="convergence_results_v3_iter${iter}"

echo "=========================================="
echo "Starting convergence computation (v3)"
echo "  Redshift: iz${iz}"
echo "  Iteration: ${iter}"
echo "  Subvolumes: ${subvols}"
echo "  Halo mass cut: 1e11 Msun"
echo "  Output: ${output_dir}"
echo "  Start time: $(date)"
echo "=========================================="

python compute_convergence_specific.py \
    --iz "${iz}" \
    --subvols "${subvols}" \
    --output-dir "${output_dir}" \
    --iteration "${iter}" \
    --mhalo-min 1e11

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "✓ Completed iz${iz} iteration ${iter}"
    echo "  End time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "⚠ Completed with errors iz${iz} iteration ${iter}"
    echo "  Exit code: ${exit_code}"
    echo "  End time: $(date)"
    echo "  Results may be partial but saved"
    echo "=========================================="
fi

exit 0
