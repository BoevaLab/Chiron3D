#!/bin/bash
# Convert a Juicer .hic file into:
# 1. POSITION_MATRIX.bed – a 5kb-binned BED file in the following format: chromosome, start, end, bin_id
# Example:
#   chromosome start   end   bin_id
#     chr1    0       5000    1
#     chr1    5000    10000   2
#     chr1    10000   15000   3
#     chr2    0       5000    49852
#     chr2    5000    10000   49853
# 2. COVERAGE_MATRIX.matrix.gz – a gzipped file in the following format bin_id_1, bin_id_2, interaction_count
# Example:
# bin_id_1  bin_id_2  interaction_count
#     3       2295    1
#     3       2868    1
#     3       11998   1
#     3       15542   1

# Update PATHS as needed
HIC_FILE="/Users/sebastian/University/Master/mt/ews-ml/src/corigami/data/contact_matrix_data/GSE108869_Control_CTCF-HiChIP_combine_allValidPairs.hic"
COOL_FILE="GSE108869_HeLaS3_CTCF_5kb.cool"
POSITION_MATRIX="GSE108869_HeLaS3_POSITION_MATRIX.bed"
COVERAGE_MATRIX="GSE108869_HeLaS3_COVERAGE_MATRIX.matrix.gz"
RESOLUTION=5000

echo "Step 1: Converting $HIC_FILE to Cooler format at ${RESOLUTION}bp resolution using Python..."
python convert_hic_to_cool.py "$HIC_FILE" "$COOL_FILE" "$RESOLUTION"
if [ $? -ne 0 ]; then
    echo "hic2cool conversion (via Python) failed."
    exit 1
fi

echo "Step 2: Dumping bins from $COOL_FILE"
cooler dump -t bins "$COOL_FILE" > bins_5kb.tsv
if [ $? -ne 0 ]; then
    echo "Failed to dump bins."
    exit 1
fi

echo "Creating POSITION_MATRIX.bed"
# Add a sequential bin ID (starting at 1) to the bins table.
awk 'BEGIN {OFS="\t"} {print $1, $2, $3, NR}' bins_5kb.tsv > "$POSITION_MATRIX"
if [ $? -ne 0 ]; then
    echo "Failed to create POSITION_MATRIX.bed."
    exit 1
fi

echo "Step 3: Dumping pixels (contacts) from $COOL_FILE"
cooler dump -t pixels "$COOL_FILE" > contacts_5kb.tsv
if [ $? -ne 0 ]; then
    echo "Failed to dump pixels."
    exit 1
fi

echo "Converting pixel indices from 0-based to 1-based and creating COVERAGE_MATRIX.tsv"
# Cooler outputs bin indices in 0-based format; add 1 to each so that they match the POSITION_MATRIX.
awk 'BEGIN {OFS="\t"} {print $1+1, $2+1, $3}' contacts_5kb.tsv > COVERAGE_MATRIX.tsv
if [ $? -ne 0 ]; then
    echo "Failed to create COVERAGE_MATRIX.tsv."
    exit 1
fi

echo "Compressing COVERAGE_MATRIX.tsv to $COVERAGE_MATRIX..."
gzip -c COVERAGE_MATRIX.tsv > "$COVERAGE_MATRIX"
if [ $? -ne 0 ]; then
    echo "Failed to compress COVERAGE_MATRIX.tsv."
    exit 1
fi

echo "Cleaning up temporary files"
rm bins_5kb.tsv contacts_5kb.tsv COVERAGE_MATRIX.tsv

echo "Conversion complete."
