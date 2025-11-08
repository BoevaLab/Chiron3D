#!/bin/bash

#SBATCH --job-name=install_pytorch
#SBATCH --output=setup_env.output.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4G

# Setup conda
source ~/.bashrc
conda activate

# Create env
conda create -y --name ews-ml python=3.10

conda activate ews-ml

pip install captum 
