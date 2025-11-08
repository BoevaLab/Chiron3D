#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --job-name=BORZOI
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/CHANGES_FULLLORA.output.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

if [ $# -eq 0 ]; then
  echo "Provide a flag name to run the script. Example: sbatch corigami-training.sh --gatc1hot"
  exit 1
fi

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

cd src/models/training
echo "directory 3:"
pwd

CONFIG_FILE="/cluster/work/boeva/shoenig/ews-ml/config/config.sh"
source "$CONFIG_FILE"

SEED=2077
FLAG=$(echo "$1" | sed 's/^--//')
echo "Using flag: $FLAG"

# Save path
BASE_SAVE_PATH="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints"
SAVE_PATH="${BASE_SAVE_PATH}/${FLAG}"
echo "Save path will be: ${SAVE_PATH}"

# Data paths
REGIONS_FILE=$REGIONS_FILE_500KB #$REGIONS_FILE_500KB #$REGIONS_FILE_1MB #REGIONS_FILE_200KB
COOL_FILE=$COOL_A673_WT 
GENOME_FEAT_PATH=$GENOM_FEAT_CTCF_A673_WT

# Model parameters
MODEL_TYPE="EnformerBackbone"
NUM_GENOM_FEAT=0

# Training Parameters
PATIENCE=7
MAX_EPOCHS=25
SAVE_TOP_N=25
NUM_GPU=4

# Dataloader Parameters
BATCH_SIZE=4
DDP_DISABLED="--ddp-disabled"
NUM_WORKERS=16

CKPT="/cluster/work/boeva/shoenig/ews-ml/training_runs_TC71_WT/checkpoints/Borzoi_FullLora2/models/epoch=7-step=8856.ckpt"

MOTIF=""  # Change this to any motif you want to use, empty string for no extra channel

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the Python script with the arguments
python3 train.py \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --regions-file $REGIONS_FILE \
  --fasta-dir $FASTA_DIR_HG19 \
  --cool-file $COOL_FILE \
  --genom-feat-path $GENOME_FEAT_PATH \
  --model-type $MODEL_TYPE \
  --num-genom-feat $NUM_GENOM_FEAT \
  --motif "$MOTIF" \
  --patience $PATIENCE \
  --max-epochs $MAX_EPOCHS \
  --save-top-n $SAVE_TOP_N \
  --num-gpu $NUM_GPU \
  --batch-size $BATCH_SIZE \
  $DDP_DISABLED \
  --num-workers $NUM_WORKERS \
  --borzoi \
  --lora \
  --use_groupnorm \
