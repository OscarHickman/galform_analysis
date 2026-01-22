#!/bin/bash
#SBATCH --job-name=conv_fine_271
#SBATCH --output=convergence_fine_iz271_%j.log
#SBATCH --error=convergence_fine_iz271_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -A durham
#SBATCH --partition=cosma5

echo "======================================================================"
echo "FINE-GRAINED CONVERGENCE COMPUTATION: iz271"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /cosma/home/durham/dc-hick2/galform_analysis

OUTPUT_DIR=${OUTPUT_DIR:-convergence_results_fine}
SUBVOLS=${SUBVOLS:-1,2,3,4,5,6,8,10,15,20,30,40,50,80,100,150,200,300,400,500,600,700,800,1000}

echo "Using OUTPUT_DIR=$OUTPUT_DIR"
echo "Subvolumes: $SUBVOLS"

# Run the computation with Python 3
python3 compute_convergence_fine.py --iz 271 --subvols "$SUBVOLS" --output-dir "$OUTPUT_DIR"

echo ""
echo "End time: $(date)"
echo "======================================================================"
