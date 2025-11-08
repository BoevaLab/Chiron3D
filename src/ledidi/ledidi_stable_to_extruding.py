import argparse
import torch
import pandas as pd
from ledidi import ledidi
from src.ledidi.my_pruning import greedy_pruning
from src.loop_calling.dataset.loop_dataset import LoopDataset
from src.models.training.train import TrainModule
from torch import nn
import os, datetime, json              
import numpy as np                      
import matplotlib
matplotlib.use("Agg")                  
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style("white")
import copy
from src.ledidi.utils import col_to_base
from src.utils import plot_modification, predict_matrix, plot_prediction
from src.ledidi.running_utils import make_loop_ends_bin_mask, process_ctcf_narrowpeak, make_only_ctcf_edit_mask, cant_edit_loop_boundaries
import torch.nn.functional as F


def has_unmappable_stripe(seq_4L: torch.Tensor,
                          min_run: int = 50,
                          region: tuple[int,int] | None = None) -> bool:
    """
    Returns True if there is a contiguous run of >= min_run unmappable columns.
    Unmappable := column sum != 1 (dead: 0/ multi-hot: >1).
    Optionally restrict to a region [lo, hi) in sequence coordinates.
    """
    assert seq_4L.ndim == 2 and seq_4L.shape[0] == 4
    L = seq_4L.shape[1] 
    lo, hi = (0, L) if region is None else region
    lo = max(0, lo); hi = min(L, hi)

    colsum = seq_4L[:, lo:hi].sum(dim=0)                # (hi-lo,)
    unmappable = (colsum != 1)                          # bool
    if unmappable.sum() == 0:
        return False

    # sliding window sum == min_run ↔ at least one run of length ≥ min_run
    k = torch.ones(min_run, device=seq_4L.device).view(1,1,-1)
    s = F.conv1d(unmappable.float().view(1,1,-1), k)
    return bool((s.squeeze(0).squeeze(0) >= min_run).any())

def _log_metrics(stage: str, wrapper: nn.Module, X: torch.Tensor) -> None:
    """
    Print corner/ratio and stripe means for the given sequence X.
    Works for both wrappers (HiChIPStripeAndCorner and HiChIPStripeAndCornerVal).
    """
    with torch.no_grad():
        y = wrapper(X).squeeze(0)  # [corner_or_ratio, mean_x, mean_y]
    corner_or_ratio = float(y[0].item())
    mean_x = float(y[1].item())
    mean_y = float(y[2].item())
    print(f"[METRICS] {stage}: corner={corner_or_ratio:.6f}  mean_x={mean_x:.6f}  mean_y={mean_y:.6f}")

    if corner_or_ratio < 3:
        return False
    return True

class HiChIPStripeAndCorner(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""

    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        ignore_k: int = 15,
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k

    def forward(self, x):
        i, j, k = self.i, self.j, self.k
        m =self.core(x)  
        m = m.squeeze()
        stripe_x = m[i, i + k : j]
        stripe_y = m[i : j - k, j - 1]

        mean_x = stripe_x.mean()
        mean_y = stripe_y.mean()

        corner_val = m[j-1, i]

        y = torch.stack([corner_val, mean_x, mean_y], dim=0)

        return y.unsqueeze(0)

def make_stable_to_extruding_loss(used_ratio, bx, by):

    biggest_stripe = max(bx, by)
    def loss_fn(y_hat: torch.Tensor, y_bar=None) -> torch.Tensor:
        y = y_hat.squeeze(0) if y_hat.dim() == 2 else y_hat  # (3,)
        corner_val, mean_x, mean_y = y[0], y[1], y[2]

        ratio_curr = corner_val / biggest_stripe

        over_desired = torch.clamp(ratio_curr - used_ratio, min=0.0)

        return 20 * over_desired
        """
        gap = (ratio_curr - used_ratio) / 0.08

        return 4 * F.softplus(gap)"""
    return loss_fn


def plot_all_for_sequence(
        element: dict,
        seq_tensor: torch.Tensor,
        core_model: nn.Module,
        base_pred: torch.Tensor,
        out_dir: str,
        tag: str,
        device: str
):
    """
    element   : deep-copied dict whose 'sequence' field matches seq_tensor
    seq_tensor: (4, L) edited sequence
    base_pred : original (unedited) prediction matrix
    tag       : 'mutation', 'undo_3', …
    """
    # 1. prediction for current sequence
    pred = predict_matrix(element, core_model, device)

    # 2a. regular prediction panel
    plot_prediction(
        element, pred,
        save_png=True,
        save_path=os.path.join(out_dir, f"{tag}_pred.png"),
        show_corner_values=True,  
        show_stripe_sums=True,   
    )

    # 2b. diff panel versus ORIGINAL baseline
    plot_modification(
        element, base_pred, pred,
        save_png=True,
        save_path=os.path.join(out_dir, f"{tag}_diff.png"),
    )


def evaluate_element(elem, core_model, device, run_dir):
    """Run LEDIDI on one loop and return a dict of metrics."""

    original_elem = copy.deepcopy(elem)
    original_sequence = elem["sequence"].to(device)

    X = elem["sequence"].unsqueeze(0).to(device)
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]
    ignore_k=15

    if has_unmappable_stripe(elem["sequence"], min_run=5000, region=[i*5000, (j-1)*5000]):
        print("Has unmappable stripe, skipping")
        return {
                "idx": int(elem.get("idx", -1)),
                "loop_start": elem.get("loop_start", None),
                "loop_end": elem.get("loop_end", None),
                "original_corner_ratio": 0,
                "edited_corner_ratio":  0,
                "original_meanX": 0,
                "edited_meanX":  0,
                "original_meanY": 0,
                "edited_meanY":  0,
                "mutations_raw": [],
                "mutations_final": [],
                "base_changes": [],
                "single_edit_contrib": None,
                "chr":   elem.get("chr", ""), 
                "contributions": [],
                "comment": "Unmappable stripet"           
            }
    
    
    core_model.eval()
    with torch.no_grad():
        base_m = core_model(X).detach()

    wrapper = HiChIPStripeAndCorner(core_model, i, j).to(device)

    with torch.no_grad():
        y0 = wrapper(X).squeeze(0)  # (3,) or 4,
    corner0 = float(y0[0].item())
    mean_x0 = float(y0[1].item())
    mean_y0 = float(y0[2].item())
    stripe0 = max(mean_x0, mean_y0)
    ratio0  = corner0 / (stripe0)
    
    max_ratio_usually = 1.2
    individual_ratio = ratio0 - 0.5 
    used_ratio = min(max_ratio_usually, individual_ratio)
    used_ratio = max(1, used_ratio)
    print(f"Used ratio: {used_ratio}")

    print(f"Corner-Val: {corner0}, Ratio: {ratio0}, Mean_X0: {mean_x0}, Mean_Y0: {mean_y0}")
    
    
    chr_dir = os.path.join(run_dir, elem["chr"]+"_YES_INPUT_MASK")
    idx_dir = os.path.join(chr_dir, str(elem["idx"]))
    os.makedirs(idx_dir, exist_ok=True)
    base_pred = predict_matrix(original_elem, core_model, device)
    plot_prediction(
        elem, base_pred,
        save_png=True,
        save_path=os.path.join(idx_dir, "initial_pred.png"),
        show_corner_values=True,  
        show_stripe_sums=True, 
    )

    loss_fn = make_stable_to_extruding_loss(used_ratio, mean_x0, mean_y0)
    input_mask = cant_edit_loop_boundaries(elem)

    X_bar = ledidi(
        model=wrapper,
        X=X,
        y_bar=torch.zeros((1, 1), device=device), 
        output_loss=loss_fn,
        l=0.08,
        tau=1.0,
        max_iter=1500,
        early_stopping_iter=1000,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
        return_history=False,
        input_mask=input_mask
    )

    _log_metrics("AFTER_EDIT_PRE_PRUNE", wrapper, X_bar)

    before_prune = X_bar.squeeze()

    # deep-copy to avoid mutating the dataset object
    before_prune_elem = copy.deepcopy(elem)
    before_prune_elem['sequence'] = before_prune

    plot_all_for_sequence(
                before_prune_elem,
                before_prune,
                core_model,
                base_pred,
                idx_dir,
                tag="before_prune",
                device=device
            )

    # ---------------------- pruning -------------------------
    updated_sequence_first = X_bar.squeeze()
    diff_mask_first = original_sequence != updated_sequence_first           

    cols_with_any_diff_first = diff_mask_first.any(dim=0)    
    diff_positions_first = torch.nonzero(cols_with_any_diff_first).flatten()
    diff_positions_first = diff_positions_first.tolist()
    print("Originall diff pos:")
    print(diff_positions_first)
    if len(diff_positions_first) > 250:
        print(f"Something went wrong, there were {len(diff_positions_first)} proposed")

        return {
                "idx": int(elem.get("idx", -1)),
                "loop_start": elem.get("loop_start", None),
                "loop_end": elem.get("loop_end", None),
                "original_corner_ratio": corner0,
                "edited_corner_ratio":  corner0,
                "original_meanX": mean_x0,
                "edited_meanX":  mean_x0,
                "original_meanY": mean_y0,
                "edited_meanY":  mean_y0,
                "mutations_raw": [],
                "mutations_final": [],
                "base_changes": [],
                "single_edit_contrib": None,
                "chr":   elem.get("chr", ""), 
                "contributions": [],
                "comment": "Weird loop, check it out"           
            }

    X_bar_p = greedy_pruning(wrapper, X, X_bar, chr_dir, idx_dir, core_model, base_pred, device, elem, threshold=1.0, verbose=True)

    _log_metrics("AFTER_PRUNE", wrapper, X_bar_p)

    updated_sequence = X_bar_p.squeeze()

    # deep-copy to avoid mutating the dataset object
    mut_elem = copy.deepcopy(elem)
    mut_elem['sequence'] = updated_sequence

    plot_all_for_sequence(
                mut_elem,
                updated_sequence,
                core_model,
                base_pred,
                idx_dir,
                tag="mutation",
                device=device
            )

    diff_mask = original_sequence != updated_sequence           

    cols_with_any_diff = diff_mask.any(dim=0)    
    diff_positions = torch.nonzero(cols_with_any_diff).flatten()
    diff_positions = diff_positions.tolist()
    raw_positions  = diff_positions.copy()

    raw_base_changes   = []
    for pos in raw_positions:
        raw_base_changes.append(
            f"{col_to_base(original_sequence[:, pos])}>"
            f"{col_to_base(updated_sequence[:, pos])}"
        )

    with torch.no_grad():
        yf = wrapper(X_bar_p).squeeze(0)  # (3,)
    corner_edit = float(yf[0].item())
    mean_xf       = float(yf[1].item())
    mean_yf       = float(yf[2].item())

    tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
    torch.save(updated_sequence.cpu(), tensor_path)  


    return {
        "idx": int(elem.get("idx", -1)),
        "loop_start": elem.get("loop_start", None),
        "loop_end": elem.get("loop_end", None),
        "original_corner_ratio": corner0,
        "edited_corner_ratio":  corner_edit,
        "original_meanX": mean_x0,
        "edited_meanX":  mean_xf,
        "original_meanY": mean_y0,
        "edited_meanY":  mean_yf,
        "mutations_raw": copy.deepcopy(raw_positions),
        "mutations_final": copy.deepcopy(diff_positions),
        "base_changes": copy.deepcopy(raw_base_changes),
        "single_edit_contrib": None,
        "chr":   elem.get("chr", ""),    
        "contributions": [],
        "comment": "Normal"
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LEDIDI edits over X or Y loops of a LoopDataset")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--csv_out", default="ledidi_eval.csv", help="Where to save results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_dir", default=None, help="Where to store plots & logs (default: ./runs/<timestamp>/)")
    
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "cmd.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = args.device

    print("Loading model")
    core_model = TrainModule.load_from_checkpoint(args.ckpt).to(device).eval()

    print("Loading dataset")

    dataset = LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/stable_extruding/stable_500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=["chr2"],
        motif="",
        use_pretrained_backbone=True
    )

    total_seen = 0
    results = []

    for idx in [4, 6, 10, 11, 14, 15, 17, 18, 19, 20, 22, 23, 25, 27, 29, 31, 32, 34, 35, 36, 38, 40, 42, 44, 45, 47, 48, 49, 51]: #range(len(dataset)):
        elem = dataset[idx]
        stripe = elem.get("status_filtered", "").strip()
        if stripe not in {"STABLE"}:
            continue  # skip non-qualifying loops
        
        total_seen += 1


        """i, j = elem["relative_loop_start"], elem["relative_loop_end"]
        M = elem["matrix"]
        if not isinstance(M, np.ndarray):
            M = M.detach().cpu().numpy()
        corner = M[i, j-1]
        horizontal = M[i, i+15:j]
        vertical = M[i:j-15, j-1]
        hmean = horizontal.mean()
        vmean = vertical.mean()
        min_str = min(hmean, vmean)
        if corner/min_str < 1.4:
            continue
        """
        
        print(f"Working with Index {idx}")
        elem["idx"] = idx  # for reporting
        metrics = evaluate_element(
            elem,
            core_model,
            device,
            run_dir, 
        )
        results.append(metrics)
        if total_seen > 5:
            print("Early stop :)")
            break

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, args.csv_out), index=False)

if __name__ == "__main__":
    main()
   
