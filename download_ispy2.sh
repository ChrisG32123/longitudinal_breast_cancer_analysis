#!/bin/bash --login
#SBATCH --job-name=ispy2_download
#SBATCH --output=/mnt/home/gerlac37/ISPY2/logs/ispy2_download_%j.log
#SBATCH --error=/mnt/home/gerlac37/ISPY2/logs/ispy2_download_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --partition=standard

#####################################################
# Run the following in the terminal:                #
# sbatch /mnt/home/gerlac37/ISPY2/download_ispy2.sh #
#                                                   #
# ISPY2 data will then be able to be found in:      #
# /mnt/scratch/gerlac37/ISPY2/data                  #
#####################################################

##############
# 1. MODULES #
##############
module purge
module load Go/1.22.1

#######################################
# 2. SET UP DIRECTORY STRUCTURE/PATHS #
#######################################
PROJECT_HOME_DIR=/mnt/home/gerlac37/ISPY2
PROJECT_SCRATCH_DIR=/mnt/scratch/gerlac37/ISPY2

LOGS=${PROJECT_HOME_DIR}/logs
DATA=${PROJECT_SCRATCH_DIR}/data
METADATA=${PROJECT_SCRATCH_DIR}/metadata

mkdir -p "$PROJECT_HOME_DIR"
mkdir -p "$PROJECT_SCRATCH_DIR"
mkdir -p "$LOGS"

if [ -d "$DATA" ]; then
  echo "Found existing data directory. Removing it..."
  rm -rf $DATA
fi

if [ -d "$METADATA" ]; then
  echo "Found existing metadata directory. Removing it..."
  rm -rf $METADATA
fi

mkdir -p "$DATA"
mkdir -p "$METADATA"

MANIFEST=${PROJECT_HOME_DIR}/ISPY2-Cohort1-inclu-ACRIN6698-full-manifest.tcia

echo "=== Check Paths ==="
echo "Manifest: $MANIFEST"
echo "Logs: $LOGS"
echo "Metadata: $METADATA"
echo "Data: $DATA"

#################################
# 3. CLEAN UP OLD FILES/REPO(s) #
#################################

# Remove old cloned repo if it exists
cd "$PROJECT_HOME_DIR"
if [ -d NBIA_data_retriever_CLI ]; then
  echo "Found existing NBIA_data_retriever_CLI directory. Removing it..."
  rm -rf NBIA_data_retriever_CLI
fi

##############################
# 4. CLONE & BUILD THE CLI   #
##############################
echo "=== Cloning NBIA_data_retriever_CLI from GitHub ==="
git clone https://github.com/ygidtu/NBIA_data_retriever_CLI.git
echo "=== Git Clone Complete ==="

cd NBIA_data_retriever_CLI
echo "=== Building the CLI Tool ==="
go mod tidy
go build -o nbia_cli .

########################################
# 5. DOWNLOAD DICOM DATA (NO PASSWORD) #
########################################
echo "=== Starting DICOM Download ==="
./nbia_cli \
  --user nbia_guest \
  --passwd "" \
  --input "$MANIFEST" \
  --output "$DATA" \
  --processes "${SLURM_CPUS_PER_TASK:-4}" \
  --debug
echo "=== Data Download Complete ==="

################################
# 6. DOWNLOAD METADATA (CLI)   #
################################
echo "=== Starting Metadata Download ==="
# The --meta flag tells NBIA_data_retriever_CLI to retrieve metadata
./nbia_cli \
  --meta \
  --user nbia_guest \
  --passwd "" \
  --input "$MANIFEST" \
  --output "$METADATA" \
  --debug
echo "=== Metadata Download Complete ==="

###################################
# 7. ALL DONE; CLEAN UP (Optional) #
###################################
cd ..
rm -rf NBIA_data_retriever_CLI  # remove the cloned repo if you don't need it anymore

echo "=== All Downloads Complete ==="
