#!/bin/bash

#SBATCH --job-name=create_env
#SBATCH --output=create_env.output.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4G

# Setup conda
source ~/.bashrc
conda activate

# Create env
conda create -y --name chiron python=3.10
conda activate chiron

pip install pandas cooler pyfaidx pyBigWig
pip install lightning lightning-bolts
pip install torch torchvision torchaudio
pip install matplotlib captum modisco-lite peft einops ipympl enformer_pytorch borzoi-pytorch


echo "Installation complete!"

conda list
