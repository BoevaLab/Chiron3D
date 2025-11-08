#!/bin/bash

#SBATCH -p compute
#SBATCH --job-name=PeakCall
#SBATCH --output=/cluster/work/boeva/shoenig/ews-ml/data/ledidi/PeakCalling.txt
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G

bdg="/cluster/work/boeva/shoenig/ews-ml/data/ledidi/fli1_jun.bedgraph"
outdir="/cluster/work/boeva/shoenig/ews-ml/data/ledidi"
cutoff="10.0"
max_gap="30"
min_len="11"
threads="4"


# Activate conda
source ~/.bashrc
conda activate ews-ml

# Make output structure
mkdir -p "${outdir}"
tmpdir="${outdir}/tmp.$$"
mkdir -p "${tmpdir}"

# Base name (no path, no extension)
base="$(basename "${bdg%.*}")"

# Paths
clean_bdg="${tmpdir}/${base}.clean.bedGraph"
sorted_bdg="${tmpdir}/${base}.sorted.bedGraph"
final_peak="${outdir}/${base}.narrowPeak"

echo "▶ Cleaning headers/comments (if any)..."
grep -v -E '^(track|browser|#)' "${bdg}" > "${clean_bdg}"

echo "▶ Sorting BedGraph..."
tmp_for_sort="${SLURM_TMPDIR:-${TMPDIR:-${tmpdir}}}"
LC_ALL=C sort --temporary-directory="${tmp_for_sort}" \
              --parallel="${threads}" \
              -k1,1 -k2,2n \
              "${clean_bdg}" -o "${sorted_bdg}"

echo "▶ Calling peaks with MACS2 bdgpeakcall..."
macs2 bdgpeakcall \
  -i "${sorted_bdg}" \
  -o "${final_peak}" \
  -c "${cutoff}" \
  -g "${max_gap}" \
  -l "${min_len}"

echo "✔ Peaks written to ${final_peak}"

# Clean up temp
rm -rf "${tmpdir}"
