#!/bin/bash

# Define variables for all arguments
SEED=2077
SAVE_PATH="checkpoints"

# Data paths
WINDOWS_FILE="/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/HeLa-S3/input_data/genomic_regions_data/local_train_hg19.bed"
FASTA_DIR="/Volumes/scratch-boeva/data/annotations/Human/hg19/chromosomes"
COOL_FILE="/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/HeLa-S3/input_data/contact_matrix_data/HeLaS3_CTCF_5000.cool"
GENOME_FEAT_PATH="/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/HeLa-S3/input_data/genomic_features_data"

# Model parameters
MODEL_TYPE="ConvTransModelSmall"
NUM_GENOM_FEAT=0

# Training Parameters
PATIENCE=1
MAX_EPOCHS=2
SAVE_TOP_N=2
NUM_GPU=1

# Dataloader Parameters
BATCH_SIZE=1
DDP_DISABLED="--ddp-disabled"
NUM_WORKERS=1

MOTIF="ATAT"  # Change this to any motif you want to use
USE_MOTIF=false  # Set to false if you don't want to use any motif

if ! $USE_MOTIF; then
  MOTIF=""
fi

# Run the Python script with the arguments
python3 train.py \
  --local \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --windows-file $WINDOWS_FILE \
  --fasta-dir $FASTA_DIR \
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
  --num-workers $NUM_WORKERS
