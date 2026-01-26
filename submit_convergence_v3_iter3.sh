#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH -J conv_iter_v3_iter3
#SBATCH -o logs/convergence_v3_iter3_iz%a_iter%A.log
#SBATCH -e logs/convergence_v3_iter3_iz%a_iter%A.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 12:00:00
#SBATCH --array=0-1

# Convergence analysis v3 iter3 only (corrected HMF/2PCF averaging) with mass cut 1e11
# Subvolumes: 1,2,4,8,10,15,20,30,50,100,200,1024

cd /cosma/home/durham/dc-hick2/galform_analysis

mkdir -p logs convergence_results_v3_iter3

module load python/3.9.19
export PYTHONPATH=$(pwd)/.venv/lib/python3.9/site-packages:$PYTHONPATH

subvols="1,2,4,8,10,15,20,30,50,100,200,1024"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    iz=271
else
    iz=207
fi

output_dir="convergence_results_v3_iter3"
iter=3

echo "=========================================="
echo "Starting convergence computation (v3 iter3)"
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
