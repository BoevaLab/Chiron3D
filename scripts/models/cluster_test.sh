#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --job-name=corigami_training
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/corigami_test.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G


# Activate conda
source ~/.bashrc
conda activate ews-ml

# Check env
echo
echo "which python"
which python


# These steps are needed to install the current package in editable mode
# so that modules in different folders can be imported properly.
# -------------------------- start
echo "directory 0:"
pwd

cd ..
echo "directory 1:"
pwd

cd ..
echo "directory 2:"
pwd

pip install -e .

cd src/corigami/training
echo "directory 3:"
pwd
# -------------------------- end


# Define variables for all arguments
SEED=2077
SAVE_PATH="/cluster/work/boeva/shoenig/ews-ml/src/corigami/checkpoints"

# Data paths
REGIONS_FILE="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/HeLaS3/genomic_regions_data/local_train_hg19.bed"
FASTA_DIR="/cluster/work/boeva/minjwang/data/hg19/chromosomes"
COOL_FILE="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/HeLaS3/contact_matrix_data/HeLaS3_CTCF_5000.cool"
GENOME_FEAT_PATH="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/HeLaS3/genomic_features_data/GSE108869_Control_CTCF_ChIPSeq_treat_fc.bw"

# Model parameters
MODEL_TYPE="ConvTransModelSmall"
NUM_GENOM_FEAT=1

# Training Parameters
PATIENCE=7
MAX_EPOCHS=10
SAVE_TOP_N=10
NUM_GPU=1

# Dataloader Parameters
BATCH_SIZE=2
DDP_DISABLED="--ddp-disabled"
NUM_WORKERS=4

# Run the Python script with the arguments
python3 train.py \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --windows-file $WINDOWS_FILE \
  --fasta-dir $FASTA_DIR \
  --cool-file $COOL_FILE \
  --genom-feat-path $GENOME_FEAT_PATH \
  --model-type $MODEL_TYPE \
  --num-genom-feat $NUM_GENOM_FEAT \
  --patience $PATIENCE \
  --max-epochs $MAX_EPOCHS \
  --save-top-n $SAVE_TOP_N \
  --num-gpu $NUM_GPU \
  --batch-size $BATCH_SIZE \
  $DDP_DISABLED \
  --num-workers $NUM_WORKERS
