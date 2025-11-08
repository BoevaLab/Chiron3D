import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.loop_calling.importance_analysis.importance_scoring import GradientScorer
from src.utils import load_model, print_element, predict_matrix, plot_prediction, load_bigwig_signal, plot_logo, plot_hic


def with_new_sequence(element: dict, onehot_seq: torch.Tensor) -> dict:
    """
    Return a deep copy of element whose sequence tensor is replaced by onehot_seq
    """
    e = copy.deepcopy(element)
    e["sequence"] = onehot_seq         
    return e


def plot_site_attributions(model: torch.nn.Module,
                           orig_element: dict,
                           updated_element: dict,
                           positions: list[int],
                           window_radius: int = 25,
                           bigwig_path: str | None = None,
                           stripe: str = "X",
                           ignore_k: int = 0,
                           device: str = "cpu"):
    """
    Parameters
    ----------
    model            : trained PyTorch model
    orig_element     : dataset element with the original sequence
    updated_element  : element whose "seq" has been replaced by the mutated one
    positions        : list of absolute genomic positions that were edited
    window_radius    : half-window size (25 ⇒ 50 bp window)
    bigwig_path      : if given, raw BigWig signal is plotted above the logos
    stripe, ignore_k, device : forwarded to GradientScorer.compute_scores
    """
    scorer = GradientScorer(device)
    # Get full-length attribution once for each sequence
    attr_orig, _ = scorer.compute_scores(model, orig_element, stripe, ignore_k)
    attr_upd,  _ = scorer.compute_scores(model, updated_element, stripe, ignore_k)
    region_start = orig_element["region_start"]   # same for both elements
    chrom        = orig_element["chr"]

    rows_per_site = 3 if bigwig_path else 2
    n_rows        = rows_per_site * len(positions)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows),
                             sharex=False, squeeze=False)
    axes = axes.flatten()  # easier indexing

    for idx, pos in enumerate(positions):
        rel = pos                      # position inside tensors
        seq_start = max(rel - window_radius, 0)
        seq_end   = min(rel + window_radius + 1, attr_orig.shape[0])  # end-exclusive
        # Optional: raw track
        if bigwig_path:
            ax_track = axes[rows_per_site * idx]
            abs_start = region_start + pos - window_radius
            abs_end   = region_start + pos + window_radius + 1
            signal = load_bigwig_signal(bigwig_path, chrom, abs_start, abs_end)
            ax_track.plot(np.arange(abs_end - abs_start), signal)
            ax_track.set_ylabel("Signal")
            ax_track.set_title(f"{chrom}:{abs_start} - {abs_end}")

        local_pos = rel - seq_start

        # Original logo
        ax_orig = axes[rows_per_site * idx + (1 if bigwig_path else 0)]
        plot_logo(attr_orig.detach().numpy(),
                  seq_start, seq_end,
                  ax=ax_orig,
                  title=f"original",
                  highlight_idx=local_pos)

        # Updated logo
        ax_upd = axes[rows_per_site * idx + (2 if bigwig_path else 1)]
        plot_logo(attr_upd.detach().numpy(),
                  seq_start, seq_end,
                  ax=ax_upd,
                  title=f"updated",
                  highlight_idx=local_pos)

        # cosmetic
        for ax in (ax_orig, ax_upd):
            ax.set_xlabel(f"±{window_radius} bp")

    plt.tight_layout()
    plt.show()


def report_nt_changes(orig_seq: torch.Tensor, updated_seq: torch.Tensor):
    """
    Print the positions where two one-hot ACGT tensors differ and
    show the change in nucleotide (e.g. 17: A → G).

    Parameters
    ----------
    orig_seq : (4, L) torch.Tensor
        One-hot encoding of the original sequence (row-order A, C, G, T).
    updated_seq : (4, L) torch.Tensor
        One-hot encoding of the updated sequence (same layout as orig_seq).

    Returns
    -------
    List[Tuple[int, str, str]]
        (position, original_base, updated_base) for every mismatch.
    """
    if orig_seq.shape != updated_seq.shape or orig_seq.shape[0] != 4:
        raise ValueError("Both tensors must be shape (4, L) in A-C-G-T order.")

    # convert one-hot columns to base indices 0-3
    orig_idx = orig_seq.argmax(dim=0)
    upd_idx  = updated_seq.argmax(dim=0)

    diff_positions = (orig_idx != upd_idx).nonzero(as_tuple=False).flatten()
    base_lookup = "ACGT"

    print(f"Total sequence-positions with any mismatch: {diff_positions.numel()}")
    if diff_positions.numel() == 0:
        return []

    changes = []
    for pos in diff_positions.tolist():
        o_base = base_lookup[orig_idx[pos]]
        u_base = base_lookup[upd_idx[pos]]
        changes.append((pos, o_base, u_base))
        print(f"{pos}: {o_base} → {u_base}")

    return changes


def apply_edits_to_sequence(element: dict,
                             edits: list[tuple[int, str, str]],
                             row_map: dict[str, int] = None) -> dict:
    """
    Return a deep copy of `element` with multiple one-hot edits applied.

    Parameters
    ----------
    element : dict
        Original data element containing a 'sequence' tensor of shape (4, L).
    edits : list of (pos, before, after)
        Positions and nucleotide changes to apply.
    row_map : dict, optional
        Mapping from nucleotide letter to row index (default A:0,C:1,G:2,T:3).

    Returns
    -------
    dict
        Deep copy of element with updated 'sequence'.
    """
    e = copy.deepcopy(element)
    seq = e["sequence"]
    if row_map is None:
        row_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    for pos, orig_base, new_base in edits:
        # Optional sanity check
        current_idx = seq[:, pos].argmax().item()
        current_base = "ACGT"[current_idx]
        if current_base != orig_base:
            raise ValueError(f"Expected base {orig_base} at pos {pos}, found {current_base}.")
        seq[:, pos].zero_()
        seq[row_map[new_base], pos] = 1

    e["sequence"] = seq
    return e


def plot_multi_site_attributions(model: torch.nn.Module,
                                 element: dict,
                                 edits: list[tuple[int, str, str]],
                                 window_radius: int = 20,
                                 bigwig_path: str | None = None,
                                 stripe: str = "X",
                                 ignore_k: int = 0,
                                 device: str = "cpu",
                                 start=None, end=None):
    """
    Compute and plot attributions over a region spanning all edits,
    highlighting each edit position within the logos.

    The y-axis range (min/max) for the logos is computed ONLY from the
    actually plotted window (orig_slice & upd_slice), not the full sequence.
    """
    # Apply all edits
    updated = apply_edits_to_sequence(element, edits)

    scorer = GradientScorer(device)
    attr_orig, _ = scorer.compute_scores(model, element, stripe, ignore_k)  # [L, C]
    attr_upd,  _ = scorer.compute_scores(model, updated, stripe, ignore_k)  # [L, C]

    # Determine region covering all edits (or honor explicit start/end)
    if (start is None or end is None):
        if not edits:
            raise ValueError("`edits` must be non-empty if start/end are not provided.")
        positions = [pos for pos, _, _ in edits]
        min_pos, max_pos = min(positions), max(positions)
        seq_len = attr_orig.shape[0]
        start_rel = max(min_pos - window_radius, 0) if start is None else start
        end_rel   = min(max_pos + window_radius + 1, seq_len) if end is None else end
    else:
        start_rel, end_rel = int(start), int(end)

    if end_rel <= start_rel:
        raise ValueError(f"Invalid window: start={start_rel}, end={end_rel}")

    # Slice attribution arrays to the plotted window
    # (use only first 4 channels A/C/G/T, consistent with plot_logo)
    orig_slice = attr_orig[start_rel:end_rel, :4]
    upd_slice  = attr_upd [start_rel:end_rel, :4]

    # --- KEY CHANGE: y-limits from the plotted window only ---
    ymin = float(torch.min(torch.min(orig_slice), torch.min(upd_slice)))
    ymax = float(torch.max(torch.max(orig_slice), torch.max(upd_slice)))

    # Compute relative highlight indices
    highlights = [pos - start_rel for pos, _, _ in edits]

    # Plotting setup
    rows = 3 if bigwig_path else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.5 * rows), sharex=False)
    axes = axes.flatten()
    ax_idx = 0

    # Optional raw signal track
    if bigwig_path:
        ax_track = axes[ax_idx]
        ax_idx += 1
        abs_start = element["region_start"] + start_rel
        abs_end   = element["region_start"] + end_rel
        signal = load_bigwig_signal(bigwig_path, element["chr"], abs_start, abs_end)
        ax_track.plot(np.arange(abs_end - abs_start), signal)
        ax_track.set_ylabel("Signal")
        ax_track.set_title(f"{element['chr']}:{abs_start}-{abs_end}")

    # Original attribution logo
    ax_orig = axes[ax_idx]
    plot_logo(orig_slice.detach().cpu().numpy(), 0, len(orig_slice),
              ax=ax_orig, title="Original", ymin=ymin, ymax=ymax)
    for hl in highlights:
        if 0 <= hl < len(orig_slice):
            ax_orig.axvline(hl, linestyle='--', linewidth=1)
    ax_idx += 1

    # Updated attribution logo
    ax_upd = axes[ax_idx]
    plot_logo(upd_slice.detach().cpu().numpy(), 0, len(upd_slice),
              ax=ax_upd, title="Updated", ymin=ymin, ymax=ymax)
    for hl in highlights:
        if 0 <= hl < len(upd_slice):
            ax_upd.axvline(hl, linestyle='--', linewidth=1)

    # Labels
    for ax in (ax_orig, ax_upd):
        ax.set_xlabel(f"Region around edits (±{window_radius} bp)")

    plt.tight_layout()
    plt.show()




import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# --- utilities ---

def per_base_change_score(attr_orig: torch.Tensor,
                          attr_upd: torch.Tensor,
                          method: str = "l1",
                          relative: bool = False,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    attr_*: [L, 4] (signed attribution per base x nucleotide)
    returns: score [L] with per-position change magnitude
    """
    diff = attr_upd - attr_orig
    if method == "l1":
        score = diff.abs().sum(dim=-1)
    elif method == "l2":
        score = torch.sqrt((diff * diff).sum(dim=-1))
    else:
        raise ValueError("method must be 'l1' or 'l2'")

    if relative:
        base_mag = 0.5 * (attr_orig.abs().sum(dim=-1) + attr_upd.abs().sum(dim=-1))
        score = score / (base_mag + eps)
    return score


def plot_single_site_attributions(model: torch.nn.Module,
                                  element: dict,
                                  min_bin: int,
                                  max_bin: int,
                                  stripe: str = "X",
                                  ignore_k: int = 0,
                                  bigwig_path: str | None = None,
                                  device: str = "cpu"):
    """
    Plot attribution/logo for a single element restricted to [min_bin, max_bin] (inclusive).
    - model, element, stripe, ignore_k, device: same semantics as before
    - min_bin, max_bin: integer inclusive indices into the attribution array
    - bigwig_path: optional path to BigWig to plot raw signal (absolute genomic coords constructed
      using element['region_start'] + bin index)
    Returns (fig, axes).
    """
    # basic validation
    if min_bin > max_bin:
        raise ValueError("min_bin must be <= max_bin")
    if min_bin < 0:
        raise ValueError("min_bin must be >= 0")

    scorer = GradientScorer(device)
    # compute attributions for the (single) element
    attr, _ = scorer.compute_scores(model, element, stripe, ignore_k)  # attr: [L,4] tensor

    seq_len = attr.shape[0]
    start = int(min_bin)
    end = int(max_bin) + 1  # make end exclusive

    # clip to available sequence
    start = max(0, start)
    end = min(seq_len, end)
    if start >= end:
        raise ValueError(f"Selected bin range [{min_bin},{max_bin}] yields empty slice after clipping to [0,{seq_len-1}]")

    slice_attr = attr[start:end]            # tensor [slice_len, 4]
    slice_np   = slice_attr.detach().cpu().numpy()

    # scalar per-base importance (L1 magnitude) for plotting above the logo
    importance = slice_attr.abs().sum(dim=-1).detach().cpu().numpy()  # shape (slice_len,)

    # plotting layout: top = optional bigwig or importance, bottom = logo
    rows = 3 if bigwig_path else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.5 * rows), sharex=False)
    axes = axes.flatten()
    ax_idx = 0

    # BigWig signal (optional)
    if bigwig_path:
        ax_bw = axes[ax_idx]
        ax_idx += 1
        abs_start = element["region_start"] + start
        abs_end   = element["region_start"] + end
        signal = load_bigwig_signal(bigwig_path, element["chr"], abs_start, abs_end)
        ax_bw.plot(np.arange(abs_end - abs_start), signal)
        ax_bw.set_ylabel("Signal")
        ax_bw.set_title(f"{element['chr']}:{abs_start}-{abs_end}")

    # Importance track (scalar)
    ax_imp = axes[ax_idx]
    ax_idx += 1
    x = np.arange(start, end) - start  # local x indices for plotting
    ax_imp.plot(x, importance, linewidth=1.5)
    ax_imp.set_xlim(x[0], x[-1])
    ax_imp.set_ylabel("L1 importance")
    ax_imp.set_title(f"Per-base importance (bins {start}:{end-1})")

    # Sequence-logo for the same slice
    ax_logo = axes[ax_idx]
    plot_logo(slice_np, 0, slice_np.shape[0], ax=ax_logo, title="Attribution logo (selected bins)")
    ax_logo.set_xlabel(f"Bins {start} .. {end-1} (absolute: {element['chr']}:{element['region_start'] + start} - {element['region_start'] + end - 1})")

    plt.tight_layout()
    return fig, axes



def smooth_1d(x: torch.Tensor, win: int = 0) -> torch.Tensor:
    """Simple moving average smoothing; win is odd kernel width."""
    if win is None or win <= 1:
        return x
    pad = win // 2
    kernel = torch.ones(win, dtype=x.dtype, device=x.device) / win
    y = torch.nn.functional.conv1d(
        x.view(1,1,-1), kernel.view(1,1,-1), padding=pad
    ).view(-1)
    return y


def topk_positions(score: torch.Tensor,
                   k: int = 5,
                   exclude: Optional[List[int]] = None,
                   exclude_margin: int = 0) -> torch.Tensor:
    """
    Returns indices of top-k positions by score, masking out a margin around edits.
    """
    s = score.clone()
    if exclude:
        for e in exclude:
            lo = max(e - exclude_margin, 0)
            hi = min(e + exclude_margin + 1, s.numel())
            s[lo:hi] = -float("inf")  # mask out
    k = min(k, (s > -float("inf")).sum().item())
    if k <= 0:
        return torch.tensor([], dtype=torch.long)
    vals, idx = torch.topk(s, k)
    # sort by genomic order for nicer plotting
    return torch.sort(idx).values


# --- your plotting with change detection ---

def plot_multi_site_attributions_with_changes(
    model: torch.nn.Module,
    element: dict,
    edits: List[Tuple[int, str, str]],
    window_radius: int = 20,
    bigwig_path: Optional[str] = None,
    stripe: str = "X",
    ignore_k: int = 0,
    device: str = "cpu",
    # new args:
    change_method: str = "l1",
    change_relative: bool = False,
    smooth_window: int = 5,
    top_k: int = 5,
    exclude_edit_margin: int = 0,
    show_delta_logos: bool = True,
    delta_logo_radius: int = 15,
    start=None,
    end=None
):
    """
    Like your function, but:
      - computes a per-base change score along the *entire* sequence
      - plots the change track and highlights the top-K biggest changes
      - optionally shows delta logos centered on those top-K changes
    """
    updated = apply_edits_to_sequence(element, edits)

    scorer = GradientScorer(device)
    attr_orig, _ = scorer.compute_scores(model, element, stripe, ignore_k)   # [L,4]
    attr_upd,  _ = scorer.compute_scores(model, updated, stripe, ignore_k)   # [L,4]

    L = attr_orig.shape[0]
    # change score across full sequence
    score = per_base_change_score(attr_orig, attr_upd,
                                  method=change_method, relative=change_relative)
    score_s = smooth_1d(score, smooth_window)

    # find top-K change loci (optionally excluding ±margin around edited bases)
    edit_positions = [pos for pos, _, _ in edits]
    tops = topk_positions(score_s, k=top_k,
                          exclude=edit_positions, exclude_margin=exclude_edit_margin)

    # region around all edits for the main logos (your original behavior)
    if start is not None:
        start_rel = start
        end_rel   = end
    elif edits:
        min_pos, max_pos = min(edit_positions), max(edit_positions)
        start_rel = max(min_pos - window_radius, 0)
        end_rel   = min(max_pos + window_radius + 1, L)
    else:
        start_rel, end_rel = 0, L

    orig_slice = attr_orig[start_rel:end_rel].detach().cpu().numpy()
    upd_slice  = attr_upd [start_rel:end_rel].detach().cpu().numpy()
    highlights = [p - start_rel for p in edit_positions if start_rel <= p < end_rel]

    rows = 3 if bigwig_path else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.6 * rows), sharex=False)
    axes = axes.ravel(); axi = 0

    if bigwig_path:
        ax_track = axes[axi]; axi += 1
        abs_start = element["region_start"] + start_rel
        abs_end   = element["region_start"] + end_rel
        signal = load_bigwig_signal(bigwig_path, element["chr"], abs_start, abs_end)
        ax_track.plot(np.arange(abs_end - abs_start), signal)
        ax_track.set_ylabel("Signal")
        ax_track.set_title(f"{element['chr']}:{abs_start}-{abs_end}")

    ax_orig = axes[axi]; axi += 1
    plot_logo(orig_slice, 0, len(orig_slice), ax=ax_orig, title="Original")
    for hl in highlights: ax_orig.axvline(hl, linestyle='--', linewidth=1)

    ax_upd = axes[axi]
    plot_logo(upd_slice, 0, len(upd_slice), ax=ax_upd, title="Updated")
    for hl in highlights: ax_upd.axvline(hl, linestyle='--', linewidth=1)

    for ax in (ax_orig, ax_upd):
        ax.set_xlabel(f"Region around edits (±{window_radius} bp)")

    plt.tight_layout()
    plt.show()

    # --- Change track over the *full* sequence ---
    plt.figure(figsize=(12, 2.5))
    xs = np.arange(L)
    plt.plot(xs, score_s.cpu().numpy(), lw=1)
    for e in edit_positions:
        plt.axvline(e, ls='--', lw=0.7, alpha=0.6)
    for t in tops.tolist():
        plt.axvline(t, ls='-', lw=1.2, alpha=0.9)
    plt.title("Per-base attribution change (smoothed)")
    plt.ylabel(f"|Δattr| ({change_method})" + (" / rel" if change_relative else ""))
    plt.xlabel("Position")
    plt.tight_layout()
    plt.show()

    # --- Optional: delta logos at the top-K change sites ---
    if show_delta_logos and len(tops) > 0:
        ncols = min(5, len(tops))
        nrows = int(np.ceil(len(tops) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.6*nrows), squeeze=False)
        for ax, pos in zip(axes.ravel(), tops.tolist()):
            s = max(pos - delta_logo_radius, 0)
            e = min(pos + delta_logo_radius + 1, L)
            # Show both updated and a signed difference logo for intuition
            plot_logo(attr_upd[s:e].detach().cpu().numpy(), 0, e - s,
                      ax=ax, title=f"Top Δ at {pos}")
            ax.axvline(pos - s, ls='--', lw=0.8)
        plt.tight_layout()
        plt.show()

    return {
        "change_score": score.detach().cpu(),
        "change_score_smooth": score_s.detach().cpu(),
        "top_change_positions": tops.detach().cpu(),
    }


def col_to_base(col: torch.Tensor) -> str:
    """
    col : (4,) one-hot column on any device
    returns e.g. 'A', 'C', …
    """
    return "ACGT"[col.argmax().item()]
