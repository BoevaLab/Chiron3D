#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=EVALUATION
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/model_evals/Aayush_WISH.txt
#SBATCH --time=02:00:00
#SBATCH --mem=64G

source ~/.bashrc
conda activate ews-ml

echo
echo "which python"
which python

cd /cluster/work/boeva/shoenig/ews-ml
pip install -e .

cd src/models/evaluation
pwd

CONFIG_FILE="/cluster/work/boeva/shoenig/ews-ml/config/config.sh"
source "$CONFIG_FILE"

REGIONS_FILE=$REGIONS_FILE_500KB #$REGIONS_FILE_200KB 
COOL_FILE=$COOL_A673_WT #$COOL_HeLaS3
GENOME_FEAT_PATH=$GENOM_FEAT_CTCF_A673_WT
NUM_GENOM_FEAT=0
OUTDIR="/cluster/work/boeva/shoenig/ews-ml/model_evals"
MODEL_TYPE="Borzoi_FULL_75"
# FULL  /cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/NEW_HOPE/models/epoch=7-step=8856.ckpt
CKPT_PATH="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/NEW_HOPE/models/epoch=7-step=8856.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/data/A673_general/CTCF_epoch=34-step=155435.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/data/A673_general/DNA_epoch=38-step=173199.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/a673/models/epoch=27-step=124348.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/NEW_HOPE/models/epoch=7-step=8856.ckpt"  #   "/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/a673/models/epoch=27-step=124348.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/NEW_HOPE/models/epoch=7-step=8856.ckpt"


python3 eval_aayush_wish.py \
  --regions-file $REGIONS_FILE \
  --fasta-dir $FASTA_DIR_HG19 \
  --cool-file $COOL_FILE \
  --genomic-feature $GENOME_FEAT_PATH \
  --num-genom-feat $NUM_GENOM_FEAT \
  --ckpt-path $CKPT_PATH \
  --out-dir $OUTDIR \
  --model-type $MODEL_TYPE \
  --borzoi
