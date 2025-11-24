#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=eval
#SBATCH --output=evaluation.output.txt
#SBATCH --time=02:00:00
#SBATCH --mem=64G

source ~/.bashrc
conda activate chiron

cd /cluster/work/boeva/shoenig/Chiron3D

pip install -e .

REGIONS_FILE="data/windows_hg19.bed"
COOL_FILE="data/A673_WT_CTCF_5000.cool"
GENOME_FEAT_PATH="data/ctcf"
FASTA_DIR_HG19="data/chromosomes"
CKPT_PATH="data/chiron-model.ckpt" 
NUM_GENOM_FEAT=0

python3 -m src.models.evaluation.evaluation \
  --regions-file $REGIONS_FILE \
  --fasta-dir $FASTA_DIR_HG19 \
  --cool-file $COOL_FILE \
  --genomic-feature $GENOME_FEAT_PATH \
  --num-genom-feat $NUM_GENOM_FEAT \
  --ckpt-path $CKPT_PATH \
  --borzoi
