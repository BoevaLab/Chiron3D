import argparse
import ast
import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.loop_calling.dataset.loop_dataset import LoopDataset
from src.models.training.train import TrainModule
from src.utils import predict_matrix

BIN_TEST_CHROMS = {
    1: ["chr1"],
    2: ["chr2", "chr8"],
    3: ["chr3", "chr12", "chr17", "chr22"],
    4: ["chr4", "chr7", "chr10", "chr15"],
    5: ["chr5", "chr6", "chr11", "chr16"],
    6: ["chr13", "chr20", "chr19"],
    7: ["chr9", "chr14", "chr18", "chr21"],
}

BASE_TRAIN_RUN_DIR = Path("/cluster/work/boeva/shoenig/ews-ml/training_runs_ledidi_A673_WT")
OUTPUT_ROOT_DIR = Path("/cluster/work/boeva/shoenig/ews-ml/output_matrices_ledidi")


def build_dataset(bin_id: int):
    """Instantiate *exactly* the dataset you used during LEDIDI optimisation."""
    return LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=BIN_TEST_CHROMS[bin_id],
        motif="",
        use_pretrained_backbone=True,
    )


def process_single_bin(bin_id: int, model: torch.nn.Module, device: torch.device):
    """Handle one bin{X}: filter CSV, run predictions, save outputs."""

    bin_dir = BASE_TRAIN_RUN_DIR / f"bin{bin_id}"
    csv_path = bin_dir / "ledidi_debug.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df["mutations_final_parsed"] = df["mutations_final"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.strip() else []
    )
    df = df[df["mutations_final_parsed"].str.len() == 1]

    dataset = build_dataset(bin_id)

    out_dir = OUTPUT_ROOT_DIR / f"bin{bin_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"bin{bin_id}"):
        idx = int(row["idx"])
        element = dataset[idx]

        torch.save(element["matrix"].cpu(), out_dir / f"matrix_{idx}.pt") 

        # 2) Prediction on the unedited sequence
        pred_ref = predict_matrix(element, model, device)
        torch.save(pred_ref, out_dir / f"pred_original_{idx}.pt")

        # 3) Prediction on the edited sequence
        chrom = row["chr"]
        best_seq_path = bin_dir / chrom / f"best_onehot_{idx}.pt"
        if not best_seq_path.exists():
            print(f"[WARN] Missing {best_seq_path}; skipping idx {idx}.")
            continue

        edited_seq = torch.load(best_seq_path, map_location="cpu")
        edited_element = copy.deepcopy(element)
        edited_element["sequence"] = edited_seq

        pred_edit = predict_matrix(edited_element, model, device)
        torch.save(pred_edit, out_dir / f"pred_edited_{idx}.pt")


def main():
    parser = argparse.ArgumentParser("Generate loop predictions for single‑edit loops across bins.")
    parser.add_argument("--bins", nargs="*", type=int, default=list(range(1, 8)), help="Which bins to process (1‑7)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="PyTorch device string")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build & freeze model once
    print("Loading model")
    CKPT="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/debug-lora-add-aug/models/epoch=14-step=16605.ckpt" #"/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Borzoi-LoraTFLAYERS/models/epoch=12-step=14391.ckpt"
    model = TrainModule.load_from_checkpoint(CKPT).to(device).eval()

    for bin_id in args.bins:
        if bin_id not in BIN_TEST_CHROMS:
            print(f"[ERROR] Unknown bin {bin_id}; skipping.")
            continue
        process_single_bin(bin_id, model, device)


if __name__ == "__main__":
    main()
