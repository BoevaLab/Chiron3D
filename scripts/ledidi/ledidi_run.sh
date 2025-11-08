#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=LEDIDI
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/training_runs_ledidi_TC71_WT/Untargeted.txt
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

CKPT="/cluster/work/boeva/shoenig/ews-ml/training_runs_TC71_WT/checkpoints/Borzoi_FullLora3/models/epoch=7-step=8856.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Borzoi-LoraTFLAYERS/models/epoch=12-step=14391.ckpt"
CSV_OUT="ledidi_debug.csv"         
N_LOOPS=500  
TAU=1.0                   
L_WEIGHT=0.02           
LOW=0.5                            
HIGH=2.0                                               
ITER=1200
LR=0.3

python -u ledidi_evaluate.py \
        --ckpt "${CKPT}" \
        --csv_out "${CSV_OUT}" \
        --l_weight "${L_WEIGHT}" \
        --low "${LOW}" \
        --high "${HIGH}" \
        --tau "${TAU}" \
        --max_iter "${ITER}" \
        --early_stop "${ITER}" \
        --lr "${LR}" \
        --device cuda \
        --run_dir /cluster/work/boeva/shoenig/ews-ml/training_runs_ledidi_TC71_WT/bin7 \
        --bin 7
echo "------------------------------------------------------------"
echo "Finished LEDIDI evaluation."
