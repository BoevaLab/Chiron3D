# Chiron3D: an interpretable deep learning framework for understanding the DNA code of chromatin looping

Chiron3D is a DNA-only deep learning model that predicts cell-type–specific CTCF HiChIP contact maps from 524,288 bp genomic windows. The model uses a frozen Borzoi backbone with lightweight adapters and a C.Origami-style pairwise head to output 105 × 105 contact matrices at 5 kb resolution.

This repository currently focuses on:

- Training Chiron3D on the A673 CTCF HiChIP dataset

- Evaluating trained checkpoints and reproducing the main quantitative and qualitative results from the figures


## Data

Chiron3D is trained on CTCF HiChIP and matched CTCF ChIP-seq from the A673 wild-type Ewing sarcoma cell line on the hg19 reference genome.

All preprocessed inputs required to run the scripts in this repository are made available via [Zenodo](https://zenodo.org/records/17741799).

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

## Setup

Download and unpack the Zenodo archive so that the repository root has the following layout:

data/
  A673_WT_CTCF_5000.cool
  borzoi/
  chiron-model.ckpt
  chromosomes/
  ctcf/
  extruding_loops.csv
  stable_loops.csv
  windows_hg19.bed

Create a Python environment and install the package in editable mode:

```bash
conda create -n chiron python=3.10
conda activate chiron
pip install -r requirements.txt
pip install -e .
```

Please note the specific requirements of  `transformers==4.50.0` and `peft==0.17.0`.

---

## Training and evaluation

Chiron3D can be run directly from the command line. All commands below are intended to be run from the repository root.

### Training from the command line

For a full run on 4 GPUs to replicate training results, run the following command:
```bash
python3 -m src.models.training.train \
  --seed 2077 \
  --save_path checkpoints \
  --regions-file data/windows_hg19.bed \
  --fasta-dir data/chromosomes \
  --cool-file data/A673_WT_CTCF_5000.cool \
  --genom-feat-path data/ctcf \
  --num-genom-feat 0 \
  --patience 7 \
  --max-epochs 25 \
  --save-top-n 25 \
  --num-gpu 4 \
  --batch-size 4 \
  --ddp-disabled \
  --num-workers 16 \
  --borzoi
  ```

This prints the best checkpoint path at the end of training.

### Evaluation

The Zenodo link already contains a pre-trained model. To replicate our evaluation results, or run another trained model checkpoint, run the following command after inserting the appropiate checkpoint location.

```bash
python3 -m src.models.evaluation.evaluation \
  --regions-file data/windows_hg19.bed \
  --fasta-dir data/chromosomes \
  --cool-file data/A673_WT_CTCF_5000.cool \
  --genomic-feature data/ctcf \
  --num-genom-feat 0 \
  --ckpt-path checkpoints/models/<checkpoint>.ckpt \
  --borzoi
```

### Optional SLURM wrappers

For SLURM-based cluster runs, example wrappers are provided in:

- scripts/model-training.sh
- scripts/model-evaluation.sh

These scripts are examples for our internal cluster setup and may need adaptation for your environment (e.g. conda environment name, absolute repository path, GPU partition, and resource requests).

### GPU configuration

In on our experiments, we used 4 × NVIDIA RTX 4090 or 3090 with 24GB of memory for training. Convergence then takes about one day. For evaluation (across three test chromosomes), 1 × NVIDIA RTX 4090 or 3090 with 24GB completes within an hour.

## Downstream Task: Loop editing

The `notebooks` folder showcases four examples of using the ledidi-based editing framework to suggest in silico edits. The outputs of the runs can be viewed in the respective notebook and the corresponding `example` folders. Please note, that the package must be installed in editable mode by running `pip install -e .` for all paths to work. On our SLURM cluster, the following command is used to run from within the `notebooks` directory:

`srun --job-name jupyter -p gpu --gres=gpu:rtx4090:1 --time 01:00:00 --cpus-per-task 16 --mem 128G bash -c 'source ~/.bashrc && conda activate chiron && cd /path/to/main/folder/Chiron3D && pip install -e . && cd notebooks && jupyter lab --ip $(hostname -i) --no-browser'`.

