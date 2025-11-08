import argparse
import torch
import pandas as pd
from ledidi import ledidi
from ledidi.pruning import greedy_pruning
from src.loop_calling.dataset.loop_dataset import LoopDataset
from src.models.training.train import TrainModule
from torch import nn
import os, datetime, json              
import numpy as np                      
import matplotlib
matplotlib.use("Agg")                  
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style("whitegrid")
import copy
from src.utils import plot_modification, predict_matrix, plot_prediction
from src.ledidi.running_utils import make_loop_ends_bin_mask, process_fl1_jun_bed_peak, make_custom_edit_mask, process_ctcf_narrowpeak, make_only_ctcf_edit_mask, make_only_ctcf_boundary_edit_mask
import torch.nn.functional as F


EPS = 1e-8  

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
    

class HiChIPStripeAndCornerVal(nn.Module):
    """Light wrapper that outputs the X-/Y-stripe ratio for one loop."""

    def __init__(
        self,
        core_model: nn.Module,
        i: int,
        j: int,
        base_m,
        ignore_k: int = 15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.core = core_model.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.i, self.j, self.k = i, j, ignore_k
        self.eps = eps
        self.base_m = base_m.squeeze(0)

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


def make_extruding_to_stable_loss(stripe_pattern, x0, y0):
    def loss_fn(y_hat: torch.Tensor, y_bar=None) -> torch.Tensor:
        y = y_hat.squeeze(0) if y_hat.dim() == 2 else y_hat  # (3,)
        corner_val, mean_x, mean_y = y[0], y[1], y[2]

        ratio_min = 1.6

        biggest_stripe = torch.max(mean_x, mean_y)
        ratio_curr = corner_val / biggest_stripe

        gap_x = ((x0 - mean_x) - 0.4) / 0.08
        gap_y = ((y0 - mean_y) - 0.4) / 0.08

        if stripe_pattern == "X":
            stripe_loss = 2 * F.softplus(gap_x)
        elif stripe_pattern == "Y":
            stripe_loss = 2 * F.softplus(gap_y)
        else:
            stripe_loss = 2 * (F.softplus(gap_x) + F.softplus(gap_y))

        gap = (ratio_min - ratio_curr) / 0.08     
        return 4 * F.softplus(gap) + stripe_loss
    return loss_fn


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


def evaluate_element(elem, core_model, device, l_weight, max_iter, early_stop, tau, lr, run_dir,
                     stripe_limit, corner_ratio_limit, stable_to_extruding, hparam=None):
    """Run LEDIDI on one loop and return a dict of metrics."""

    print(elem)
    original_elem = copy.deepcopy(elem)
    original_sequence = elem["sequence"].to(device)

    X = elem["sequence"].unsqueeze(0).to(device)
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]
    ignore_k=15

    core_model.eval()
    with torch.no_grad():
        base_m = core_model(X).detach()

    wrapper = HiChIPStripeAndCornerVal(core_model, i, j, base_m=base_m).to(device)

    with torch.no_grad():
        y0 = wrapper(X).squeeze(0)  # (3,) or 4,
    corner0, mean_x0, mean_y0 = float(y0[0]), float(y0[1]), float(y0[2])
    stripe0 = max(mean_x0, mean_y0)
    ratio0  = corner0 / (stripe0 + EPS)

    # use ratio0 in logs and gating
    print(f"[METRICS] BASELINE: ratio={ratio0:.3f} corner={corner0:.3f} mean_x={mean_x0:.3f} mean_y={mean_y0:.3f}")

    already_ok = (ratio0 >= 1.6)

    if already_ok:
        os.makedirs(os.path.join(run_dir, elem["chr"]), exist_ok=True)         
        print(f"✓ Loop idx {elem.get('idx', -1)} already inside target band "
              f"; skipping optimisation.")
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
            "comment": "Not prominent enough"                      
        }
    
    chr_dir = os.path.join(run_dir, elem["chr"])
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

    stripe_pattern = elem.get("status_filtered", "").strip()
    loss_fn = make_extruding_to_stable_loss(stripe_pattern=stripe_pattern, x0=mean_x0, y0=mean_y0)

    ctcf_dict = process_ctcf_narrowpeak()
    input_mask = make_custom_edit_mask(elem, ctcf_dict, intersect_with_boundaries=False)

    X_bar = ledidi(
        model=wrapper,
        X=X,
        y_bar=torch.zeros((1, 1), device=device), 
        output_loss=loss_fn,
        l=l_weight,
        tau=tau,
        max_iter=1500,
        early_stopping_iter=early_stop,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=lr,
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
    
    torch.save(X.cpu(), os.path.join(chr_dir, f"original_seq_{elem['idx']}.pt"))
    torch.save(X_bar.cpu(), os.path.join(chr_dir, f"updated_seq_{elem['idx']}.pt"))

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
            "comment": "Not prominent enough"                      
        }  

def main():
    parser = argparse.ArgumentParser(description="Evaluate LEDIDI edits over X or Y loops of a LoopDataset")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--csv_out", default="ledidi_eval.csv", help="Where to save results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l_weight", type=float, default=2.0, help="Weight on edit penalty (λ)")
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--early_stop", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=None, help="Process only first N qualifying loops (debug)")
    parser.add_argument("--run_dir", default=None, help="Where to store plots & logs (default: ./runs/<timestamp>/)")
    parser.add_argument("--stripe_limit", type=float, default=0.9,
    help="Upper band for per-bin stripe mean (use your stable mean; can tune).")
    parser.add_argument("--corner_ratio_limit", type=float, default=1.5,
        help="Required minimum corner/neighbor mean ratio.")
    parser.add_argument("--k_corner", type=int, default=1,
        help="Neighborhood half-width for corner background (k=1 => 3x3 minus center).")
    parser.add_argument("--ignore_k", type=int, default=15,
        help="Bins to skip nearest the diagonal for stripe computation.")
    parser.add_argument("--w_stripe", type=float, default=1.0,
        help="Weight for stripe-suppression loss.")
    parser.add_argument("--w_corner", type=float, default=1.0,
    help="Weight for corner-enhancement loss.")
    parser.add_argument("--stable_to_extruding",action="store_true",help="Use the reverse objective (stable → extruding). "
    "If omitted, defaults to extruding → stable."
    )
    parser.add_argument("--hparam",type=float, default=1.0)
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
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=["chr1"],
        motif="",
        use_pretrained_backbone=True
    )

    total_seen = 0
    results = []

    #for idx in range(1,10):
    #43, 104, 63, 62, 132
    #for idx in [6, 13, 29, 32, 35, 39]: # stable
    #for idx in range(len(dataset)): # extruding to stable
    for idx in [43,104,63,62,132]:
        #for idx in range(1):
        #idx = 32
        elem = dataset[idx]

        stripe = elem.get("status_filtered", "").strip()
        if stripe not in {"X", "Y", "X,Y"}:
            continue  # skip non-qualifying loops
        
        if elem["loop_end"] - elem["loop_start"] < 300_000:
            continue

        i, j = elem["relative_loop_start"], elem["relative_loop_end"]
        M = elem["matrix"]
        if not isinstance(M, np.ndarray):
            M = M.detach().cpu().numpy()
        corner = M[i, j-1]
        horizontal = M[i, i+15:j]
        vertical = M[i:j-15, j-1]
        hmean = horizontal.mean()
        vmean = vertical.mean()
        max_str = max(hmean, vmean)
        if corner/max_str > 1.4:
            continue
        
        total_seen += 1
        
        print(f"Working with Index {idx}")
        elem["idx"] = idx  # for reporting
        metrics = evaluate_element(
            elem,
            core_model,
            device,
            args.l_weight,
            args.max_iter,
            args.early_stop,
            args.tau,
            args.lr,
            run_dir, 
            stripe_limit=args.stripe_limit,
            corner_ratio_limit=args.corner_ratio_limit,
            stable_to_extruding=args.stable_to_extruding,
            hparam=args.hparam
        )
        results.append(metrics)
        if total_seen % 50 == 0:
            print(f"Processed {total_seen} qualifying loops …")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, args.csv_out), index=False)

if __name__ == "__main__":
    main()
   
