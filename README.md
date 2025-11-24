# Chiron3D: an interpretable deep learning framework for understanding the DNA code of chromatin looping

Chiron3D is a DNA-only deep learning model that predicts cell-type–specific CTCF HiChIP contact maps from 524,288 bp genomic windows. The model uses a frozen Borzoi backbone with lightweight adapters and a C.Origami-style pairwise head to output 105 × 105 contact matrices at 5 kb resolution.

This repository currently focuses on:

- Training Chiron3D on the A673 CTCF HiChIP dataset

- Evaluating trained checkpoints and reproducing the main quantitative and qualitative results from the figures


## Data

Chiron3D is trained on CTCF HiChIP and matched CTCF ChIP-seq from the A673 wild-type Ewing sarcoma cell line on the hg19 reference genome.

All preprocessed inputs required to run the scripts in this repository will be made available via Zenodo:

TODO: add Zenodo DOI / URL here once uploaded.

### Expected directory layout

After downloading and unpacking the Zenodo archive, the repository expects:

```
data/
  A673_WT_CTCF_5000.cool      # 5 kb binned CTCF HiChIP contact map (hg19)
  windows_hg19.bed            # 524,288 bp windows tiled with 50 kb stride
  chiron-model.ckpt           # pretrained checkpoint for evaluation only
  chromosomes/                # hg19 FASTA files per chromosome (e.g. chr1.fa, ...)
  ctcf/                       # CTCF feature tracks (ChIP-seq)
  borzoi/                     # Borzoi model directory
```

## Training and evaluation

There are two slurm scripts provided in the `scripts` directory: `model-evaluation.sh` and `model-training.sh`. For training, Chiron3D requires 4 × NVIDIA RTX 4090 or 3090 with 24GB of memory. Training takes about 20 hours. For evaluation, 1 × NVIDIA RTX 4090 or 3090 with 24GB finishes in 30 minutes.

## Loop editing visualizations (before/after plots)

To reproduce the before/after contact map plots for in silico loop edits (used in the figures), this repository provides the original and edited DNA sequences to explore in the `notebooks` folder.

