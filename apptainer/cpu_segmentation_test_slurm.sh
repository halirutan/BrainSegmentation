#!/bin/bash -l
#SBATCH --job-name=segmenation
#SBATCH --output=segmentation_log_%j.txt
#SBATCH --error=segmentation_err_%j.txt
#SBATCH -D /ptmp/pscheibe/models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:05:00
#SBATCH --mem=0

module purge
module load apptainer

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

apptainer exec -B /ptmp/pscheibe/models mpcdf_raven.sif bash -c "/ptmp/pscheibe/models/segmentation_test_cpu.sh"

