# Chiron3D: an interpretable deep learning framework for understanding the DNA code of chromatin looping

Chiron3D is a DNA-only deep learning model that predicts cell-type–specific CTCF HiChIP contact maps from 524,288 bp genomic windows. The model uses a frozen Borzoi backbone with lightweight adapters and a C.Origami-style pairwise head to output 105 × 105 contact matrices at 5 kb resolution.

This repository currently focuses on:

- Training Chiron3D on the A673 CTCF HiChIP dataset

- Evaluating trained checkpoints and reproducing the main quantitative and qualitative results from the figures


## Data

Chiron3D is trained on CTCF HiChIP and matched CTCF ChIP-seq from the A673 wild-type Ewing sarcoma cell line on the hg19 reference genome.

All preprocessed inputs required to run the scripts in this repository are made available via [Zenodo](https://zenodo.org/records/17655272?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY5NTkzNWY3LTY3YzEtNGY1Ni1hZTRiLTA5MzVmMzg4Mjc4MyIsImRhdGEiOnt9LCJyYW5kb20iOiIxNDU4YjE2YmIxZDg0NjYyN2FjMjgzZjZkMmUzYjU3NSJ9.g895dn6RGbVtzIs351GTNvhYAfJZa8Tt4pKz1LRgP8KZwMMtEagMOWxr9CNJzXHSA2-NwcaEeAcSr64pODzizA).

## Setup

Download and unpack the Zenodo archive and have the following layout in the top level directory.

```
data/
  A673_WT_CTCF_5000.cool      # 5 kb binned CTCF HiChIP contact map (hg19)
  borzoi/                     # Weights of backbone from borzoi-pytorch
  chiron-model.ckpt           # pretrained checkpoint for evaluation + downstream 
  chromosomes/                # hg19 FASTA files per chromosome (e.g. chr1.fa, ...)
  ctcf/                       # CTCF feature track (ChIP-seq)
  extruding_loops.csv         # Dataset of extruding loops classified by Tweed
  stable_loops.csv            # Dataset of stable loops classified by FitHiChIP
  windows_hg19.bed            # 524,288 bp windows tiled with 50 kb stride
```

Create a conda environment and install all required packages. For a SLURM-cluster, there is an install script already provided in `scripts/create_environment.sh`. Please note the specific requirements of  `transformers==4.50.0` and `peft==0.17.0`.

## Training and evaluation

There are two slurm scripts provided in the `scripts` directory: `model-evaluation.sh` and `model-training.sh`. For training, Chiron3D requires 4 × NVIDIA RTX 4090 or 3090 with 24GB of memory. Training takes about one day. For evaluation (across three test chromosomes), 1 × NVIDIA RTX 4090 or 3090 with 24GB finishes in 30 minutes.

## Downstream Task: Loop editing

The `notebooks` folder showcases four examples of using the ledidi-based editing framework to suggest in silico edits. The outputs of the runs can be viewed in the respective notebook and the corresponding `example` folders. Please note, that the package must be installed in editable mode by running `pip install -e .` for all paths to work. On our SLURM cluster, the following command is used to run from within the `notebooks` directory: 

`srun --job-name jupyter -p gpu --gres=gpu:rtx4090:1 --time 01:00:00 --cpus-per-task 16 --mem 128G bash -c 'source ~/.bashrc && conda activate chiron && cd /path/to/main/folder/Chiron3D && pip install -e . && cd notebooks && jupyter lab --ip $(hostname -i) --no-browser'`. 

