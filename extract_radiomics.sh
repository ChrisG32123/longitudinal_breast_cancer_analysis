#!/bin/bash --login
#SBATCH --job-name=ispy2_radiomics
#SBATCH --output=/mnt/home/gerlac37/ISPY2/logs/patient_radiomics_v3/radiomics_%A_%a.out
#SBATCH --error=/mnt/home/gerlac37/ISPY2/logs/patient_radiomics_v3/radiomics_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-113

module purge
module load Miniforge3
conda activate ispy2

# create the directory named with this job’s ID
LOG_DIR=/mnt/home/gerlac37/ISPY2/logs/patient_radiomics_v3

python /mnt/home/gerlac37/ISPY2/process_radiomics.py \
    --patient-index "${SLURM_ARRAY_TASK_ID}"
