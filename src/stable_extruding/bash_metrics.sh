#!/bin/bash
#SBATCH -p compute
#SBATCH --job-name=LoopPred
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/src/stable_extruding/log.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=128G

# ─── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate ews-ml

echo
echo "Python executable:"
which python
echo

# ─── Project install (editable) ────────────────────────────────────────────────
cd /cluster/work/boeva/shoenig/ews-ml
pip install -e .

cd src/stable_extruding
echo "directory 3:"
pwd

python -u gather_metrics.py

echo "Finished"
