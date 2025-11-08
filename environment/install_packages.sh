#!/bin/bash
#SBATCH --job-name=install_packages
#SBATCH --output=install_packages.output.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G

# Setup conda
source ~/.bashrc
conda activate ews-ml

pip install pandas cooler pyfaidx pyBigWig
pip install lightning lightning-bolts
pip install torch torchvision torchaudio
pip install matplotlib captum modisco-lite peft enformer_pytorch einops ipympl


echo "Installation complete!"

conda list
