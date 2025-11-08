#!/bin/bash

#SBATCH --job-name=combine_npy_files
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/logs/loop_calling/combine_npy_files2.out.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --export=ALL

# Activate conda environment
source ~/.bashrc
conda activate ews-ml

: "${CELL:?CELL not set}"
: "${CASE:?CASE not set}"
: "${SAVEDIR:?ERROR: SAVEDIR not set. sbatch --export=SAVEDIR=ATAC,...}"

# Define the directory containing the .npy files and the Python script location
PROJECT_ROOT=/cluster/work/boeva/shoenig/ews-ml
DATAPATH="$PROJECT_ROOT/prelim_results/loop_calling/$CELL/$SAVEDIR"
SCRIPT_PATH="/cluster/work/boeva/shoenig/ews-ml/src/loop_calling/importance_analysis/combine_split_importance_scores.py"

# Change to the directory where the .npy files are located
cd "$DATAPATH" || { echo "Error: Cannot change directory to $DATAPATH"; exit 1; }

echo "Running Python script to combine .npy files..."
python "$SCRIPT_PATH"

# Check if the Python script ran successfully
if [ $? -eq 0 ]; then
    echo "Combination of npy files completed successfully!"
else
    echo "Error: Combining npy files failed."
    exit 1
fi
