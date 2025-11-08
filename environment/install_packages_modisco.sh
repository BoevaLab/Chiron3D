#!/bin/bash
#SBATCH --job-name=install_packages
#SBATCH --output=install_packages.output.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G

# Setup conda
source ~/.bashrc
conda activate ews-tfmodisco

pip install pandas numpy
pip install modisco-lite
conda install -y -c bioconda meme


echo "Installation complete!"

conda list

