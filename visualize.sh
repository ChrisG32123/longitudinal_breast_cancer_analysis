#!/bin/bash --login
#SBATCH --job-name=viz_seg
#SBATCH --output=/mnt/home/gerlac37/ISPY2/logs/viz_seg.log
#SBATCH --error=/mnt/home/gerlac37/ISPY2/logs/viz_seg.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=standard

module purge
module load Miniforge3
conda activate ispy2

echo "=== Visualizing Segmentation Overlays ==="

python /mnt/home/gerlac37/ISPY2/src/visualize_segmentations.py

echo "=== Creating GIFs ==="

python /mnt/home/gerlac37/ISPY2/src/make_gifs.py

echo "Done!"
