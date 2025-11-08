# Common utility functions shared by the different Ledidi scripts (that all are trying to optimize different objectives)
import torch
from torch import nn
import os
import pandas as pd
from src.utils import predict_matrix, plot_prediction, plot_modification
import numpy as np 



def col_to_base(col: torch.Tensor) -> str:
    """
    col : (4,) one-hot column on any device
    returns e.g. 'A', 'C', …
    """
    return "ACGT"[col.argmax().item()]


def plot_all_for_sequence(
        element: dict,
        seq_tensor: torch.Tensor,
        core_model: nn.Module,
        base_pred: torch.Tensor,
        out_dir: str,
        tag: str,
        device: str,
        show_corner_values=False,  
        show_stripe_sums=False,   
):
    """
    element   : deep-copied dict whose 'sequence' field matches seq_tensor
    seq_tensor: (4, L) edited sequence
    base_pred : original (unedited) prediction matrix
    tag       : 'mutation', 'undo_3', …
    """
    pred = predict_matrix(element, core_model, device)

    plot_prediction(
        element, pred,
        save_png=True,
        save_path=os.path.join(out_dir, f"{tag}_pred.png"),
        show_corner_values=show_corner_values,  
        show_stripe_sums=show_stripe_sums,   
    )

    plot_modification(
        element, base_pred, pred,
        save_png=True,
        save_path=os.path.join(out_dir, f"{tag}_diff.png"),
    )


def process_ctcf_bed_peak():
    peaks = pd.read_csv(
        "/cluster/work/boeva/shoenig/ews-ml/data/A673_general/CTCF_bed/GSM3901157_A351C25_narrow_peaks_clean.bed.gz",
        sep="\t",
        compression="gzip",
        header=None,
        names = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak"
    ])

    peaks_by_chr = {
        chrom: grp[["start", "end"]].values  
        for chrom, grp in peaks.groupby("chrom", sort=False)
    }

    return peaks_by_chr


def process_ctcf_narrowpeak(path="/cluster/work/boeva/shoenig/ews-ml/data/A673_general/CTCF_bed/GSM3901157_A351C25_narrow_peaks_clean.bed.gz", summit_half_width=50):
    """
    Read ENCODE narrowPeak and return dict[chrom] -> np.ndarray of shape (N, 2)
    with [start, end] intervals centered on the summit (± summit_half_width).
    Skips rows with undefined summit (peak == -1).
    """
    cols = ["chrom", "start", "end", "name", "score", "strand",
            "signalValue", "pValue", "qValue", "peak"]
    peaks = pd.read_csv(path, sep="\t", compression="gzip", header=None, names=cols)

    # Keep only rows with a defined summit
    peaks = peaks[peaks["peak"] >= 0].copy()

    # Compute summit position and build tight intervals
    summit = peaks["start"].to_numpy(np.int64) + peaks["peak"].to_numpy(np.int64)
    start = (summit - summit_half_width).astype(np.int64)
    end   = (summit + summit_half_width).astype(np.int64)

    # Ensure start < end (just in case) and pack per chromosome
    intervals = pd.DataFrame({"chrom": peaks["chrom"], "start": start, "end": end})
    # Optionally drop any degenerate intervals
    intervals = intervals[intervals["end"] > intervals["start"]]

    peaks_by_chr = {
        chrom: grp[["start", "end"]].to_numpy(dtype=np.int64)
        for chrom, grp in intervals.groupby("chrom", sort=False)
    }
    return peaks_by_chr


def process_spklf_bed_peak():
    peaks = pd.read_csv(
    "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/sp_klf.narrowPeak",
    sep="\t",
    compression="infer",
    header=None,
    names = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak"
    ],  
    dtype={
        "chrom": "string",
        "start": "Int64",
        "end": "Int64",
        "name": "string",
        "score": "Int64",
        "strand": "string",
        "signalValue": "float64",
        "pValue": "float64",
        "qValue": "float64",
        "peak": "Int64"
    })

    peaks_by_chr = {
        chrom: grp[["start", "end"]].values  
        for chrom, grp in peaks.groupby("chrom", sort=False)
    }

    return peaks_by_chr


def process_fl1_jun_bed_peak():
    peaks = pd.read_csv(
    "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/fli1_jun.narrowPeak",
    sep="\t",
    compression="infer",
    header=None,
    names = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak"
    ],  
    dtype={
        "chrom": "string",
        "start": "Int64",
        "end": "Int64",
        "name": "string",
        "score": "Int64",
        "strand": "string",
        "signalValue": "float64",
        "pValue": "float64",
        "qValue": "float64",
        "peak": "Int64"
    })

    peaks_by_chr = {
        chrom: grp[["start", "end"]].values  
        for chrom, grp in peaks.groupby("chrom", sort=False)
    }

    return peaks_by_chr


def make_ctcf_edit_mask(elem, ctcf_dict, flank=10, device="cpu"):
    """
    Returns a BoolTensor (L,) where True = LOCK (do not edit).
    """
    chrom = elem["chr"]

    mask = torch.zeros(524288, dtype=torch.bool, device=device)

    if chrom not in ctcf_dict:
        return mask        

    # Absolute coordinates of current sequence
    seq_start = elem["region_start"]     
    seq_end   = elem["region_end"]

    peaks = ctcf_dict[chrom]

    # -------- vectorised overlap test ----------
    # keep peaks whose end >= (seq_start-flank)  AND  start <= (seq_end+flank)
    overlaps = peaks[(peaks[:, 1] >= seq_start - flank) &
                     (peaks[:, 0] <= seq_end   + flank)]

    # mark every overlapping interval
    for start, end in overlaps:
        rel_start = max(0, start - seq_start - flank)
        rel_end   = min(524288, end   - seq_start + flank)
        mask[rel_start:rel_end] = True

    return mask


def make_only_ctcf_edit_mask(elem, ctcf_dict, flank=10, device="cpu", no_boundaries=False):
    """
    Returns a BoolTensor (L,) where True = LOCK (do not edit).
    """
    chrom = elem["chr"]

    mask = torch.ones(524288, dtype=torch.bool, device=device)

    if chrom not in ctcf_dict:
        return mask        

    # Absolute coordinates of current sequence
    seq_start = elem["region_start"]     
    seq_end   = elem["region_end"]

    peaks = ctcf_dict[chrom]

    # -------- vectorised overlap test ----------
    # keep peaks whose end >= (seq_start-flank)  AND  start <= (seq_end+flank)
    overlaps = peaks[(peaks[:, 1] >= seq_start - flank) &
                     (peaks[:, 0] <= seq_end   + flank)]

    # mark every overlapping interval
    for start, end in overlaps:
        rel_start = max(0, start - seq_start - flank)
        rel_end   = min(524288, end   - seq_start + flank)
        mask[rel_start:rel_end] = False

    if no_boundaries:
        for bin_idx in {elem["relative_loop_start"], elem["relative_loop_end"]}:
            start = max(0, bin_idx * 5000)
            end   = min(524288, start + 5000)
            mask[start:end] = 1

    return mask

def make_only_ctcf_boundary_edit_mask(elem, ctcf_dict, flank=10, device="cpu"):
    """
    Returns a BoolTensor (L,) where True = LOCK (do not edit).
    """
    chrom = elem["chr"]

    mask = torch.zeros(524288, dtype=torch.bool, device=device)

    if chrom not in ctcf_dict:
        return mask        

    # Absolute coordinates of current sequence
    seq_start = elem["region_start"]     
    seq_end   = elem["region_end"]

    peaks = ctcf_dict[chrom]

    # -------- vectorised overlap test ----------
    # keep peaks whose end >= (seq_start-flank)  AND  start <= (seq_end+flank)
    overlaps = peaks[(peaks[:, 1] >= seq_start - flank) &
                     (peaks[:, 0] <= seq_end   + flank)]

    # mark every overlapping interval
    for start, end in overlaps:
        rel_start = max(0, start - seq_start - flank)
        rel_end   = min(524288, end   - seq_start + flank)
        mask[rel_start:rel_end] = True
    
    mask2 = torch.ones(524288, dtype=torch.bool, device=device)
    for bin_idx in {elem["relative_loop_start"], elem["relative_loop_end"]}:
        start = max(0, bin_idx * 5000)
        end   = min(524288, start + 5000)
        mask2[start:end] = True

    mask = mask & mask2

    mask = ~mask

    print(mask.sum()) # expected to be around 524288

    return mask


def make_custom_edit_mask(elem, custom_dict, flank=10, device="cpu", intersect_with_boundaries=False):
    """
    Returns a BoolTensor (L,) where True = LOCK (do not edit).
    If intersect_with_boundaries=True, only zero out regions that intersect
    both custom_dict peaks and loop boundary bins.
    """
    L = elem["sequence"].shape[-1]
    mask = torch.ones(L, dtype=torch.bool, device=device)
    chrom = elem["chr"]

    if chrom not in custom_dict:
        return mask        

    seq_start = elem["region_start"]     
    seq_end   = elem["region_end"]
    peaks = custom_dict[chrom]

    # Keep peaks that overlap this sequence (flank-aware)
    overlaps = peaks[(peaks[:, 1] >= seq_start - flank) &
                     (peaks[:, 0] <= seq_end   + flank)]

    # Create base "custom" mask
    custom_mask = torch.ones(L, dtype=torch.bool, device=device)
    for start, end in overlaps:
        rel_start = max(0, start - seq_start - flank)
        rel_end   = min(L, end - seq_start + flank)
        custom_mask[rel_start:rel_end] = 0

    if intersect_with_boundaries:
        # Generate loop boundary mask
        loop_mask = torch.ones(L, dtype=torch.bool, device=device)
        for bin_idx in {elem["relative_loop_start"], elem["relative_loop_end"]}:
            start = max(0, bin_idx * 5000)
            end   = min(L, start + 5000)
            loop_mask[start:end] = 0

        # Only unlock positions where both masks are unlocked (logical AND on editable regions → OR on lock masks)
        mask = ~(~custom_mask & ~loop_mask)
    else:
        mask = custom_mask

    return mask



def make_non_anchor_bin_mask(elem, stripe, device="cpu"):
    """
    Return BoolTensor (L,) where True = LOCK (can't edit).
    Only the 5kb bin at the *non-anchored* loop boundary is editable.
    Assumes elem['relative_loop_start'] / ['relative_loop_end'] are 5kb bin indices.
    """
    L = elem["sequence"].shape[-1]
    mask = torch.ones(L, dtype=torch.bool, device=device)  

    i = elem["relative_loop_start"]
    j = elem["relative_loop_end"]

    if stripe == "X":
        bin_idx = j - 1   
    else:  
        bin_idx = i       

    start = max(0, bin_idx * 5000)
    end   = min(L, start + 5000)

    mask[start:end] = False  # unlock this 
    return mask


def make_loop_ends_bin_mask(elem, device="cpu"):
    """
    Return BoolTensor (L,) where True = LOCK (can't edit).
    Only the 5kb bins at the two loop boundaries are editable.
    """
    L = elem["sequence"].shape[-1]
    mask = torch.ones(L, dtype=torch.bool, device=device)  

    i = elem["relative_loop_start"]
    j = elem["relative_loop_end"]

    for bin_idx in {i, j}:
        start = max(0, bin_idx * 5000)
        end   = min(L, start + 5000)
        if start < L:                      
            mask[start:end] = False
    return mask


def cant_edit_loop_boundaries(elem, device="cpu"):
    """
    Lock boundaries
    """
    L = elem["sequence"].shape[-1]
    mask = torch.zeros(L, dtype=torch.bool, device=device)  

    i = elem["relative_loop_start"]
    j = elem["relative_loop_end"]
          
    mask[elem["relative_loop_start"]*5000:(elem["relative_loop_start"]+1)*5000] = True
    mask[(elem["relative_loop_end"]-1)*5000:elem["relative_loop_end"]*5000] = True
    return mask
