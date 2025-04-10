#!/bin/bash --login
#SBATCH --job-name=ispy2_clean
#SBATCH --output=/mnt/home/gerlac37/ISPY2/logs/ispy2_clean_%j.log
#SBATCH --error=/mnt/home/gerlac37/ISPY2/logs/ispy2_clean_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --partition=standard

module purge
module load Miniforge3
conda activate ispy2

PROJECT_SCRATCH_DIR=/mnt/scratch/gerlac37/ISPY2

# Test directories
# INPUT_DIR=${PROJECT_SCRATCH_DIR}/test_organized_data
# OUTPUT_DIR=${PROJECT_SCRATCH_DIR}/test_cleaned_data

INPUT_DIR=${PROJECT_SCRATCH_DIR}/data
OUTPUT_DIR=${PROJECT_SCRATCH_DIR}/data_cleaned

mkdir -p "$OUTPUT_DIR"

echo "=== Cleaning Data ==="

# Reorganize and clean the data
python /mnt/home/gerlac37/ISPY2/src/clean_data.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --nprocs ${SLURM_CPUS_PER_TASK}

echo "Done!"
