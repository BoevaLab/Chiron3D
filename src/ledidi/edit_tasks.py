import os
import copy
from typing import Dict, Any, Tuple, List, Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F

from ledidi import ledidi
from src.ledidi.custom_pruning import greedy_pruning, PruningConfig
from src.ledidi.utils import col_to_base, make_intra_loop_mask
from src.ledidi.wrappers import StripeWrapper, RatioWrapper, HiChIPStripeAndCorner
from src.ledidi.losses import stripe_diff_loss, ratio_inverted_ballpark_loss, make_extruding_to_stable_loss, make_stable_to_extruding_loss


# helpers
def prepare_loop(
    elem: Dict[str, Any],
    device: str,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, int, int]:
    """
    Deep-copy elem and return:
      elem_copy, sequence(4,L), X(1,4,L), (i,j) loop indices.
    """
    elem = copy.deepcopy(elem)
    seq = elem["sequence"].to(device)        # (4, L)
    X = seq.unsqueeze(0)                     # (1, 4, L)
    i = int(elem["relative_loop_start"])
    j = int(elem["relative_loop_end"])
    return elem, seq, X, i, j


def ensure_chr_dir(run_dir: str, chrom: str, idx: int) -> str:
    chr_dir = os.path.join(run_dir, chrom)
    os.makedirs(chr_dir, exist_ok=True)
    return chr_dir


def get_diff_positions(
    seq_orig: torch.Tensor,  # (4, L)
    seq_new: torch.Tensor,   # (4, L)
) -> List[int]:
    diff_mask = (seq_orig != seq_new).any(dim=0)
    return torch.nonzero(diff_mask, as_tuple=False).flatten().tolist()


def compute_base_changes(
    seq_orig: torch.Tensor,
    seq_new: torch.Tensor,
    positions: List[int],
) -> List[str]:
    return [
        f"{col_to_base(seq_orig[:, pos])}>{col_to_base(seq_new[:, pos])}"
        for pos in positions
    ]

def has_unmappable_stripe(seq_4L: torch.Tensor,
                          min_run: int = 50) -> bool:
    """
    Returns True if there is a contiguous run of >= min_run unmappable columns.
    """
    assert seq_4L.ndim == 2 and seq_4L.shape[0] == 4

    colsum = seq_4L.sum(dim=0)                
    unmappable = (colsum != 1)                         
    if unmappable.sum() == 0:
        return False
    
    k = torch.ones(min_run, device=seq_4L.device).view(1,1,-1)
    s = F.conv1d(unmappable.float().view(1,1,-1), k)
    return bool((s.squeeze(0).squeeze(0) >= min_run).any())


# score functions used by pruning
def stripe_score_fn(y_full: torch.Tensor, y_mod: torch.Tensor) -> torch.Tensor:
    return (y_full.squeeze() - y_mod.squeeze()).abs()


def ratio_score_fn(y_full: torch.Tensor, y_mod: torch.Tensor) -> torch.Tensor:
    return (y_full.squeeze() - y_mod.squeeze()).abs()


def ratio_score_stable_to_extr(
    y_full: torch.Tensor, y_mod: torch.Tensor
) -> torch.Tensor:
    cf, mxf, myf = y_full.squeeze(0)
    cm, mxm, mym = y_mod.squeeze(0)
    ratio_full = cf / torch.max(mxf, myf)
    ratio_mod = cm / torch.max(mxm, mym)
    return ratio_mod - ratio_full


def ratio_score_extr_to_stable(
    y_full: torch.Tensor, y_mod: torch.Tensor
) -> torch.Tensor:
    cf, mxf, myf = y_full.squeeze(0)
    cm, mxm, mym = y_mod.squeeze(0)
    ratio_full = cf / torch.max(mxf, myf)
    ratio_mod = cm / torch.max(mxm, mym)
    return ratio_full - ratio_mod


def run_ledidi_with_pruning(
    wrapper: nn.Module,
    X: torch.Tensor,                 
    seq_orig: torch.Tensor,          
    device: str,
    loss_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ledidi_kwargs: Dict[str, Any],
    input_mask: Optional[torch.Tensor],
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pruning_cfg: PruningConfig,
) -> torch.Tensor:
    """
    Shared core:
      1) run Ledidi with given wrapper + loss_fn
      2) check number of edited positions
      3) greedy pruning
    Returns updated sequence (4,L) or None if aborted.
    """
    X_bar = ledidi(
        model=wrapper,
        X=X,
        y_bar=torch.zeros((1, 1), device=device),
        output_loss=loss_fn,
        input_mask=input_mask,
        return_history=False,
        **ledidi_kwargs,
    )
    seq_bar = X_bar.squeeze(0)            # (4, L)
    diff_positions_first = get_diff_positions(seq_orig, seq_bar)

    if len(diff_positions_first) > 200:
        print(f"[LEDIDI] too many edits ({len(diff_positions_first)} > {200}); aborting.")
        return None

    X_bar_p = greedy_pruning(
        model=wrapper,
        X_orig=X,
        X_edited=X_bar,
        score_fn=score_fn,
        cfg=pruning_cfg,
    )
    seq_final = X_bar_p.squeeze(0)
    return seq_final

# Evaluation functions

def evaluate_asym_to_sym(
    elem: Dict[str, Any],
    core_model: nn.Module,
    device: str,
    stripe: str,
    run_dir: str,
    pruning_threshold: float = 5.0,
) -> None:
    """
    Stripe edit using StripeWrapper. Writes best_onehot_{idx}.pt into run_dir/chr.
    """
    elem, seq_orig, X, i, j = prepare_loop(elem, device)
    if has_unmappable_stripe(elem["sequence"]):
        print("unmappable stripe; skipping.")
        return
    ignore_k = 15

    with torch.no_grad():
        m0 = core_model(X)
        sum_x0 = m0[:, i, i + ignore_k : j].sum(dim=-1).item()
        sum_y0 = m0[:, i : j - ignore_k, j - 1].sum(dim=-1).item()

    wrapper = StripeWrapper(
        core_model, i, j,
        stripe=stripe,
        base_sum_x=sum_x0,
        base_sum_y=sum_y0,
    ).to(device)

    # dynamic target band
    with torch.no_grad():
        if stripe == "X":
            dynamic_thresh = 0.3 * sum_x0
        else:
            dynamic_thresh = 0.3 * sum_y0

        y_hat = wrapper(X)
        diff = float(y_hat.item())

        init_loss = float(stripe_diff_loss(y_hat, thresh=dynamic_thresh).item())
        print(f"[STRIPE] diff={diff:.3f}, dyn_thresh={dynamic_thresh:.3f}, loss={init_loss:.3f}")

    # early exit
    if abs(diff) < dynamic_thresh:
        print(f"[STRIPE] loop {elem.get('idx', -1)} inside target band; skipping.")
        return

    ledidi_kwargs = dict(
        l=0.08,
        tau=1.0,
        max_iter=1500,
        early_stopping_iter=1200,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
    )
    pruning_cfg = PruningConfig(threshold=pruning_threshold, min_remaining=1, verbose=True)

    def loss_fn(y_hat: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        return stripe_diff_loss(y_hat, thresh=dynamic_thresh)

    seq_final = run_ledidi_with_pruning(
        wrapper=wrapper,
        X=X,
        seq_orig=seq_orig,
        device=device,
        loss_fn=loss_fn,
        ledidi_kwargs=ledidi_kwargs,
        score_fn=stripe_score_fn,
        pruning_cfg=pruning_cfg,
        input_mask=None,
    )
    if seq_final is None:
        return

    chr_dir = ensure_chr_dir(run_dir, elem["chr"], elem["idx"])
    tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
    torch.save(seq_final.cpu(), tensor_path)
    print(f"[STRIPE] saved {tensor_path}")


def evaluate_sym_to_asym(
    elem: Dict[str, Any],
    core_model: nn.Module,
    device: str,
    stripe: str,
    run_dir: str,
    pruning_threshold: float = 0.2,
) -> None:
    """
    Symmetric→asymmetric stripe edit using RatioWrapper.
    """
    print("[SYM→ASYM] start")
    elem, seq_orig, X, i, j = prepare_loop(elem, device)
    if has_unmappable_stripe(elem["sequence"]):
        print("unmappable stripe; skipping.")
        return

    wrapper = RatioWrapper(core_model, i, j, stripe=stripe).to(device)

    with torch.no_grad():
        y_hat = wrapper(X)
        original_ratio = float(y_hat.item())
        init_loss = float(ratio_inverted_ballpark_loss(y_hat, low=0.5, high=2.0).item())
        print(f"[SYM→ASYM] ratio={original_ratio:.3f}, loss={init_loss:.3f}")

    if init_loss < 0.5:
        print(f"[SYM→ASYM] loop {elem.get('idx', -1)} inside target band; skipping.")
        return

    ledidi_kwargs = dict(
        l=0.02,
        tau=1.0,
        max_iter=1500,
        early_stopping_iter=1000,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
    )
    pruning_cfg = PruningConfig(threshold=pruning_threshold, min_remaining=1, verbose=True)

    def loss_fn(y_hat: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        return ratio_inverted_ballpark_loss(y_hat, original_ratio)

    seq_final = run_ledidi_with_pruning(
        wrapper=wrapper,
        X=X,
        seq_orig=seq_orig,
        device=device,
        loss_fn=loss_fn,
        ledidi_kwargs=ledidi_kwargs,
        input_mask=None,
        score_fn=ratio_score_fn,
        pruning_cfg=pruning_cfg,
    )
    if seq_final is None:
        return

    chr_dir = ensure_chr_dir(run_dir, elem["chr"], elem["idx"])
    tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
    torch.save(seq_final.cpu(), tensor_path)
    print(f"[SYM→ASYM] saved {tensor_path}")


def evaluate_stable_to_extruding(
    elem: Dict[str, Any],
    core_model: nn.Module,
    device: str,
    run_dir: str,
    pruning_threshold: float = 0.35,
    max_initial_edits: int = 250,
) -> None:
    elem, seq_orig, X, i, j = prepare_loop(elem, device)
    if has_unmappable_stripe(elem["sequence"]):
        print("unmappable stripe; skipping.")
        return

    wrapper = HiChIPStripeAndCorner(core_model, i, j).to(device)
    with torch.no_grad():
        y0 = wrapper(X).squeeze(0)
    corner0 = float(y0[0].item())
    mean_x0 = float(y0[1].item())
    mean_y0 = float(y0[2].item())
    stripe0 = max(mean_x0, mean_y0)
    ratio0 = corner0 / stripe0
    used_ratio = ratio0 - 0.5
    print(f"[STABLE→EXTR] corner={corner0:.3f}, ratio={ratio0:.3f}, used_ratio={used_ratio:.3f}")

    loss_fn = make_stable_to_extruding_loss(used_ratio, mean_x0, mean_y0)
    input_mask = make_intra_loop_mask(elem, device)

    ledidi_kwargs = dict(
        l=0.02,
        tau=1.0,
        max_iter=1500,
        early_stopping_iter=1000,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
    )
    pruning_cfg = PruningConfig(threshold=pruning_threshold, min_remaining=1, verbose=True)

    seq_final = run_ledidi_with_pruning(
        wrapper=wrapper,
        X=X,
        seq_orig=seq_orig,
        device=device,
        loss_fn=loss_fn,
        ledidi_kwargs=ledidi_kwargs,
        input_mask=input_mask,
        score_fn=ratio_score_stable_to_extr,
        pruning_cfg=pruning_cfg,
        max_initial_edits=max_initial_edits,
    )
    if seq_final is None:
        return

    chr_dir = ensure_chr_dir(run_dir, elem["chr"], elem["idx"])
    tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
    torch.save(seq_final.cpu(), tensor_path)
    print(f"[STABLE→EXTR] saved {tensor_path}")


def evaluate_extruding_to_stable(
    elem: Dict[str, Any],
    core_model: nn.Module,
    device: str,
    run_dir: str,
    pruning_threshold: float = 0.35,
) -> None:
    elem, seq_orig, X, i, j = prepare_loop(elem, device)
    if has_unmappable_stripe(elem["sequence"]):
        print("unmappable stripe; skipping.")
        return

    with torch.no_grad():
        base_m = core_model(X).detach()

    wrapper = HiChIPStripeAndCorner(core_model, i, j, base_m=base_m).to(device)
    with torch.no_grad():
        y0 = wrapper(X).squeeze(0)
    corner0 = float(y0[0].item())
    mean_x0 = float(y0[1].item())
    mean_y0 = float(y0[2].item())
    stripe0 = max(mean_x0, mean_y0)
    ratio0 = corner0 / stripe0
    ratio_min = ratio0 + 0.5
    print(f"[EXTR→STABLE] corner={corner0:.3f}, ratio={ratio0:.3f}, ratio_min={ratio_min:.3f}")

    loss_fn = make_extruding_to_stable_loss(mean_x0, mean_y0, ratio_min)
    input_mask = make_intra_loop_mask(elem, device)

    ledidi_kwargs = dict(
        l=0.08,
        tau=1.0,
        max_iter=1500,
        early_stopping_iter=1000,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
    )
    pruning_cfg = PruningConfig(threshold=pruning_threshold, min_remaining=1, verbose=True)

    seq_final = run_ledidi_with_pruning(
        wrapper=wrapper,
        X=X,
        seq_orig=seq_orig,
        device=device,
        loss_fn=loss_fn,
        ledidi_kwargs=ledidi_kwargs,
        input_mask=input_mask,
        score_fn=ratio_score_extr_to_stable,
        pruning_cfg=pruning_cfg,
    )
    if seq_final is None:
        return

    chr_dir = ensure_chr_dir(run_dir, elem["chr"], elem["idx"])
    tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
    torch.save(seq_final.cpu(), tensor_path)
    print(f"[EXTR→STABLE] saved {tensor_path}")
