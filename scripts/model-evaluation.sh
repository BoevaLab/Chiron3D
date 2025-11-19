#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --job-name=eval
#SBATCH --output=evaluation.output.txt
#SBATCH --time=02:00:00
#SBATCH --mem=64G

source ~/.bashrc
conda activate chiron

pip install -e .

cd src/models/evaluation

REGIONS_FILE="../../data/windows_hg19.bed"
COOL_FILE="../../data/A673_WT_CTCF_5000.cool"
OUTDIR="/cluster/work/boeva/shoenig/ews-ml/model_evals"
GENOME_FEAT_PATH="../../data/ctcf"
FASTA_DIR_HG19="../../data/chromosomes"
CKPT_PATH="../../data/chiron-model.ckpt" 

python3 evaluation.py \
  --regions-file $REGIONS_FILE \
  --fasta-dir $FASTA_DIR_HG19 \
  --cool-file $COOL_FILE \
  --genomic-feature $GENOME_FEAT_PATH \
  --num-genom-feat $NUM_GENOM_FEAT \
  --ckpt-path $CKPT_PATH \
  --borzoi
