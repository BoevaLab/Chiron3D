#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=create
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/data/ledidi/log.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G


# Activate conda
source ~/.bashrc
conda activate ews-ml

# Check env
echo
echo "which python"
which python


# Install package
cd /cluster/work/boeva/shoenig/ews-ml
pip install -e .

cd src/ledidi
echo "directory 3:"
pwd

export PYTHONUNBUFFERED=1

python -u create_bedGraph_for_Motifs.py
        
echo "------------------------------------------------------------"
echo "Finished LEDIDI evaluation."
