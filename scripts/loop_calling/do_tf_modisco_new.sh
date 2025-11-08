#!/bin/bash

#SBATCH --job-name=tfmodisco_run
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/logs/loop_calling/tfmodisco_run-full-new.out.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --export=ALL

# Activate conda environment
source ~/.bashrc

CONFIG_FILE="/cluster/work/boeva/shoenig/ews-ml/config/config.sh"
source "$CONFIG_FILE"

: "${CELL:?CELL not set}"
: "${CASE:?CASE not set}"
: "${SAVEDIR:?ERROR: SAVEDIR not set. sbatch --export=SAVEDIR=ATAC,...}"
: "${SCORING:?SCORING not set (e.g. sbatch --export=SCORING=gradient,…)}"


STABLE="${STABLE:-false}"


# Define the directory containing the .npy files and the Python script location
PROJECT_ROOT=/cluster/work/boeva/shoenig/ews-ml
DATAPATH="$PROJECT_ROOT/prelim_results/loop_calling/$CELL/$SAVEDIR"
RESULTS_PATH="$PROJECT_ROOT/results/loop_calling/$CELL/$SAVEDIR/$SCORING"

mkdir -p "${RESULTS_PATH}"
method="$SCORING"
NUM_SEQ=25000  # Maximum number of seqlets

# Define the lists of stripes and boundaries
if [[ "$STABLE" == "true" ]]; then
  stripes=("STABLE")                  
else
  stripes=("sym" "asym")              
fi
boundaries=("anchor" "non_anchor")

for stripe in "${stripes[@]}"; do
  for boundary in "${boundaries[@]}"; do
    if [[ "$stripe" == "STABLE" ]]; then
        NUM_SEQ=30000
    elif [[ "$stripe" == "X" || "$stripe" == "Y" ]]; then
      NUM_SEQ=15000       
    else                   
      NUM_SEQ=80000
    fi
    # Define file paths with DATAPATH prepended
    conda activate ews-ml
    OHE_FILE="${DATAPATH}/Imp_Stripe_${stripe}_Method_${method}_sequences_${boundary}.npy"  # Path to the one-hot encoded file
    SCORES_FILE="${DATAPATH}/Imp_Stripe_${stripe}_Method_${method}_scores_${boundary}.npy"  # Path to the scores file
    H5_FILE="${DATAPATH}/Imp_Stripe_${stripe}_Method_${method}_modisco_results_${boundary}_${NUM_SEQ}.h5"  # Path to the output file

    # Run the modisco motifs command
    echo "=== Running MoDISco for stripe=${stripe}, boundary=${boundary} ==="
    modisco motifs -s "$OHE_FILE" -a "$SCORES_FILE" -n "$NUM_SEQ" -o "$H5_FILE" -w 5000 -l 3

    if [ $? -eq 0 ]; then
      echo "  ✓ Completed: results in $H5_FILE"
    else
      echo "  ✗ FAILED for stripe=${stripe}, boundary=${boundary}"
      exit 1
    fi

    conda activate ews-tfmodisco-cpython
    REPORT_DIR="${RESULTS_PATH}/report_${method}_${boundary}_${stripe}_${NUM_SEQ}"  # Path to store the report
    ZIP_FILE="${RESULTS_PATH}/report_${method}_${boundary}_${stripe}_${NUM_SEQ}"  # Name of the zip file
    mkdir -p "$REPORT_DIR"

    echo "Generating MoDISco report..."
    modisco report -i "$H5_FILE" -o "$REPORT_DIR/" -s "$REPORT_DIR/" -m "$MOTIFS_MEME_FILE"

    if [ $? -ne 0 ]; then
      echo "ERROR: MoDISco report failed for stripe=${stripe}, boundary=${boundary}"
      exit 1
    fi
    echo "✔ Report generated in $REPORT_DIR/"

    echo "[zip]  compressing $REPORT_DIR → $ZIP_FILE"
    zip -r "$ZIP_FILE" "$REPORT_DIR"
    rm -rf "$REPORT_DIR"

    # Check if zipping was successful
    if [ $? -eq 0 ]; then
        echo "Report folder successfully compressed into $ZIP_FILE"
    else
        echo "Error: Failed to compress the report folder."
        exit 1
    fi
  done
done

echo "All MoDISco jobs finished."
