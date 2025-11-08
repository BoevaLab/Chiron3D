#!/bin/bash

#SBATCH --job-name=tfmodisco_preparation
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/logs/loop_calling/tfmodisco_preparation_%a.out.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --array=0-9   # Will be overwritten by script

DELAY=$((SLURM_ARRAY_TASK_ID * 30))
echo "Waiting for $DELAY seconds before starting job ${SLURM_ARRAY_TASK_ID}..."
sleep $DELAY

# Activate conda environment
source ~/.bashrc
conda activate ews-ml

CONFIG_FILE="/cluster/work/boeva/shoenig/ews-ml/config/config.sh"
source "$CONFIG_FILE"

# — require essential env vars or exit —
: "${CELL:?ERROR: CELL not set. sbatch --export=CELL=HeLaS3,...}"
: "${CASE:?ERROR: CASE not set. sbatch --export=CASE=BASE,...}"
: "${SAVEDIR:?ERROR: SAVEDIR not set. sbatch --export=SAVEDIR=ATAC,...}"
: "${SCORING:?ERROR: SCORING not set. sbatch --export=SCORING=deeplift,...}"
: "${STRIPE:?ERROR: STRIPE not set. sbatch --export=STRIPE=X,...}"
: "${USE_CHUNK:?ERROR: USE_CHUNK not set. sbatch --export=USE_CHUNK=1,...}"
: "${ARRAY_LEN:?ERROR: ARRAY_LEN not set. sbatch --export=TOTAL_LEN=4154,...}"
: "${NUM_CHUNKS:?ERROR: NUM_CHUNKS not set. sbatch --export=NUM_CHUNKS=10,...}"
: "${USE_MOTIF:?ERROR: USE_MOTIF not set. sbatch --export=USE_MOTIF=true,...}"
: "${MOTIF:?ERROR: MOTIF not set. sbatch --export=MOTIF=GATC,...}"
: "${NUM_GENOM_FEATS:?ERROR: NUM_GENOM_FEATS not set. sbatch --export=NUM_GENOM_FEATS=2,...}"


STABLE="${STABLE:-false}"

PROJECT_ROOT=/cluster/work/boeva/shoenig/ews-ml
cd "$PROJECT_ROOT"

# ensure clean, fresh editable install
pip install --upgrade --force-reinstall -e .

VAR_W="WEIGHTS_${CELL}_${SAVEDIR}"
WEIGHTS_PATH="${!VAR_W}"

COOL_VAR="COOL_${CELL}"
COOL_FILE="${!COOL_VAR}"

FEAT_VAR="GENOM_FEAT_CTCF_${CELL}"
GENOM_FEAT_PATH="${!FEAT_VAR}"

if [[ "$STABLE" == "true" ]]; then
  LOOPS_VAR="STABLE_LOOPS_${CELL}"  
else
  LOOPS_VAR="LOOPS_${CELL}"          
fi

WINDOWS_FILE="${!LOOPS_VAR}"

FASTA_DIR="$FASTA_DIR_HG19"
BLACKLIST_FILE="$BLACKLIST_HG19"

BASE_SAVE_DIR="$PROJECT_ROOT/prelim_results/loop_calling/$CELL/$SAVEDIR"
mkdir -p "${BASE_SAVE_DIR}"

if [ "$STRIPE" = "X,Y" ] || [ "$STRIPE" = "STABLE" ]; then
  SAVE_NAME="$BASE_SAVE_DIR/${SLURM_ARRAY_TASK_ID}_Imp"
else
  SAVE_NAME="$BASE_SAVE_DIR/Imp"
fi

# Number of chunks (should match the array range)
if [ "$USE_CHUNK" -eq 1 ]; then
  CHUNK_SIZE=$(( (ARRAY_LEN + NUM_CHUNKS - 1) / NUM_CHUNKS ))
  START_INDEX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
  END_INDEX=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE ))
  if [ "$END_INDEX" -gt "$ARRAY_LEN" ]; then
      END_INDEX=$ARRAY_LEN
  fi
  echo "Processing chunk: indices ${START_INDEX} to ${END_INDEX}"
  CHUNK_ARGS="--start-index ${START_INDEX} --end-index ${END_INDEX}"
else
  CHUNK_ARGS=""
fi

if ! $USE_MOTIF; then
  MOTIF=""
fi

if [[ "$BORZOI" == "true" ]]; then
  BORZOI_FLAG="--borzoi"
else
  BORZOI_FLAG=""
fi


# Run the Python script with the specified arguments
python3 -m src.loop_calling.importance_analysis.run_scoring \
  --scoring ${SCORING} \
  --stripe ${STRIPE} \
  --blacklist-file ${BLACKLIST_FILE} \
  --weights-path ${WEIGHTS_PATH} \
  --windows-file ${WINDOWS_FILE} \
  --fasta-dir ${FASTA_DIR} \
  --cool-file ${COOL_FILE} \
  --genom-feat-path ${GENOM_FEAT_PATH} \
  --save-name ${SAVE_NAME} \
  --motif "${MOTIF}" \
  --num-genom-feats ${NUM_GENOM_FEATS} \
  ${CHUNK_ARGS} \
  ${ENFORMER_FLAG} \
  ${BORZOI_FLAG}
