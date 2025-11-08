#!/bin/bash
#SBATCH --job-name=master-loop-workflow
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/logs/loop_calling/pipeline.out.txt
#SBATCH --time=1-00:00:00

set -euo pipefail

# common exports
export CELL=TC71_WT
export CASE=BASE
export SAVEDIR=BORZOI_FULLLORA2 #BORZOI_FULLLORA
export USE_MOTIF=false
export MOTIF=GATC
export SCORING=gradient
export ARRAY_LEN=13193 #5978 #8452 #5978
export NUM_GENOM_FEATS=0 #0
export BORZOI=true

SCRIPT_DIR="/cluster/work/boeva/shoenig/ews-ml/scripts/loop_calling"
LOGDIR="/cluster/work/boeva/shoenig/ews-ml/logs/loop_calling/${CELL}/${SAVEDIR}"

mkdir -p "${LOGDIR}"
declare -a JOBS

# 1) dispatch X, Y, then X,Y scoring array‐jobs
for STRIPE in X Y X,Y; do
  if [[ "$STRIPE" == "X,Y" ]]; then
    export USE_CHUNK=1
    export NUM_CHUNKS=10

  else
    export USE_CHUNK=0
    export NUM_CHUNKS=1
  fi
  export STRIPE

  echo "→ submitting scoring for STRIPE=$STRIPE (chunks=$NUM_CHUNKS)…"
  JOB_ID=$(sbatch --parsable \
    --array=0-$((NUM_CHUNKS-1)) \
    --output="${LOGDIR}/tfmodisco_preparation_${STRIPE}_%a.out.txt" \
    "$SCRIPT_DIR/do_importance_scoring.sh")
  echo "   got job $JOB_ID"
  JOBS+=("$JOB_ID")

  sleep 60
done

# build dependency list: e.g. 123:124:125
ALL_DEP=$(IFS=:; echo "${JOBS[*]}")

# 2) combine .npy (after all scoring arrays complete)
echo "→ combining .npy files (afterok:$ALL_DEP)…"
COMBINE_JOB=$(sbatch --parsable \
  --dependency=afterok:$ALL_DEP \
  "$SCRIPT_DIR/combine_split_importance_scores.sh")
echo "   combine job $COMBINE_JOB"

#3) MoDISco run (after combine)
echo "→ running MoDISco (afterok:$COMBINE_JOB)…"
FINAL_JOB=$(sbatch --parsable \
  --dependency=afterok:$COMBINE_JOB \
  "$SCRIPT_DIR/do_tf_modisco.sh")
echo "   final MoDISco job $FINAL_JOB"

