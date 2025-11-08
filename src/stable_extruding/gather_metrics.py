from src.loop_calling.dataset.loop_dataset import LoopDataset
from src.utils import load_model, print_element, predict_matrix, plot_prediction, load_bigwig_signal, plot_logo, plot_hic, plot_modification
from src.models.training.train import TrainModule
from peft import PeftModel, LoraConfig
from src.loop_calling.importance_analysis.importance_scoring import GradientScorer, calculate_input_x_gradient, IntegratedGradientsScorer
import pandas as pd
import torch
import numpy as np
import pickle
import pyfaidx
import pyBigWig as pbw
from tqdm import tqdm


extruding_dataset = LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="all",
        val_chroms=[],
        test_chroms=["chr2"],
        motif="",
        use_pretrained_backbone=True
    )


stable_dataset = LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/stable_extruding/stable_500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="all",
        val_chroms=[],
        test_chroms=["chr2"],
        motif="",
        use_pretrained_backbone=True
    )



corners_extr = []
means_extr = []
medians_extr = []
lengths_extr = []

for idx in tqdm(range(len(extruding_dataset))):
    elem = extruding_dataset[idx]
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]

    M = elem["matrix"]
    if not isinstance(M, np.ndarray):
        M = M.detach().cpu().numpy()
    corner = M[i, j-1]
    horizontal = M[i, i+15:j]
    vertical = M[i:j-15, j-1]
    hmean = horizontal.mean()
    vmean = vertical.mean()
    hmedian = np.median(horizontal)
    vmedian = np.median(vertical)
    corners_extr.append(corner)
    means_extr.append(max(hmean, vmean))
    medians_extr.append(max(hmedian, vmedian))
    lengths_extr.append(j-i)



corners_stable = []
means_stable = []
medians_stable = []
lengths_stable = []

for idx in tqdm(range(len(stable_dataset))):
    elem = stable_dataset[idx]
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]
    M = elem["matrix"]
    if not isinstance(M, np.ndarray):
        M = M.detach().cpu().numpy()
    corner = M[i, j-1]
    horizontal = M[i, i+15:j]
    vertical = M[i:j-15, j-1]
    
    hmean = horizontal.mean()
    vmean = vertical.mean()
    hmedian = np.median(horizontal)
    vmedian = np.median(vertical)
    corners_stable.append(corner)
    means_stable.append(max(hmean, vmean))
    medians_stable.append(max(hmedian, vmedian))
    lengths_stable.append(j-i)

np.savez_compressed(
    "all_loop_stats.npz",
    corners_extr=np.asarray(corners_extr, dtype=float),
    means_extr=np.asarray(means_extr, dtype=float),
    medians_extr=np.asarray(medians_extr, dtype=float),
    lengths_extr=np.asarray(lengths_extr, dtype=int),
    corners_stable=np.asarray(corners_stable, dtype=float),
    means_stable=np.asarray(means_stable, dtype=float),
    medians_stable=np.asarray(medians_stable, dtype=float),
    lengths_stable=np.asarray(lengths_stable, dtype=int),
)
