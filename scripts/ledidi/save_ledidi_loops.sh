#!/bin/bash
#
# loop_pred.slurm
#
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=LoopPred
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/output_matrices_ledidi/loop_pred.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
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

cd src/ledidi
echo "directory 3:"
pwd

python -u save_all_loops.py 

echo "Finished loop-matrix predictions."
