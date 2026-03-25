import torch
import matplotlib.pyplot as plt
import os
from huggingface_hub import snapshot_download
import pyBigWig
import numpy as np
from src.loop_calling.importance_analysis.importance_scoring import GradientScorer
import pandas as pd
import logomaker
import matplotlib.patches as patches
import pyfaidx
import cooler
import numpy as np
import pandas as pd
import os
from src.models.dataset.utils import onehotencode_dna

def load_model(model, weights_path, device=torch.device("cpu")):
    """
    Loads the model weights onto the specified device.

    Args:
        model: The PyTorch model architecture.
        weights_path (str): Path to the checkpoint file containing the weights.
        device (torch.device): cuda or cpu

    Returns:
        model: The model loaded with the checkpoint weights.
    """

    print("Loading weights...")
    print("Loading model to:", device)

    model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model_weights = checkpoint['state_dict']

    new_state_dict = {}
    for key in model_weights:
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = model_weights[key]

    model.load_state_dict(new_state_dict)

    return model


def print_element(element):

    print_order = [
        "sequence", "features", "matrix", "chrom", "region_start", "region_end",
        "loop_start", "loop_end", "relative_loop_start", "relative_loop_end", "enrichX", "enrichY", "enrich_status"
    ]

    for key in print_order:
        if key in element:
            value = element[key]
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")


def predict_matrix(element, model, device):
    """
    Prepares the input tensor from the given element, performs prediction using the model,
    and returns the processed output.

    Args:
        element (dict): Contains 'sequence' and optionally 'features'.
        model (torch.nn.Module): The model to use for prediction.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The processed output from the model (on cpu).
    """
    # Prepare input tensor
    test_input = element["sequence"]
    if "features" in element:
        test_input = torch.cat((test_input, element["features"]), dim=0)

    test_input = test_input.unsqueeze(0)  # Add batch dimension
    # Perform prediction
    with torch.no_grad():
        test_input = test_input.to(device)
        output = model(test_input)
        output = output.squeeze(0).cpu()
        output = torch.relu(output)

    return output


# TODO If needed!
"""def predict_seq(model, chr, start, end, channels, with_ctcf, cool_path):
    FASTA_DIR="/cluster/work/boeva/minjwang/data/hg19/chromosomes"
    fasta = pyfaidx.Fasta(f"{FASTA_DIR}/{chr}.fa")
    sequence = onehotencode_dna(fasta[chr][start:end].seq, channels)

    if with_ctcf:
        cool = cooler.Cooler(cool_path)
        test_input = torch.cat((sequence, ))
"""



def plot_hic(element):
    """
    Plots the Hi-C matrix from `element` with loop positions overlaid.
    """
    # extract matrix and determine global vmin/vmax
    mat = element["matrix"]
    arr = mat.detach().cpu().numpy() if isinstance(mat, torch.Tensor) else mat
    vmin, vmax = arr.min(), arr.max()

    # set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_title("CTCF HiChIP Matrix")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Bin")

    # overlay loop corners
    rs = element["relative_loop_start"]
    re = element["relative_loop_end"] - 1  # make end inclusive
    ax.scatter([rs, re, re], [rs, rs, re],
               color="green", s=20, marker="o",
               label="Loop corners")
    ax.legend(loc="upper right")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Interaction Count")

    #plt.tight_layout()
    plt.show()



import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

# ────────────────────────────────────────────────────────────
def _format_kb(n: int) -> str:
    """Return a genomic coordinate in whole kilobases, e.g. 173 890 kb."""
    return f"{n // 1_000:,}".replace(",", "\u202f") + " kb"   # thin-space separator


def _stripe_ratio(mat: torch.Tensor,
                  element: dict,
                  stripe: str = "X",
                  ignore_k: int = 15) -> float:
    """
    Compute the ratio of stripe sums
    """
    i = element["relative_loop_start"]
    j = element["relative_loop_end"]

    sum_x = mat[i, i + ignore_k : j].sum()
    sum_y = mat[i : j - ignore_k, j - 1].sum()

    # guard against division by 0:
    if stripe.upper() == "X":
        return (sum_x / sum_y).item() if sum_y != 0 else float("nan")
    else:
        return (sum_y / sum_x).item() if sum_x != 0 else float("nan")
    

def plot_prediction(
    element: dict,
    output,
    index: int = 0,
    save_png: bool = False,
    corigami_loops: bool = False,
    gene_csv: str = "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/gencode_processed.csv",
    ignore_k=15,
    save_path: str | None = None,
    show_corner_values: bool = False,  
    show_stripe_sums: bool = False,   
):
    """
    Plots predicted and experimental Hi-C matrices plus a gene-track panel.

    `element` must now carry:
        - matrix                (N×N)
        - relative_loop_start
        - relative_loop_end
        - chr                   e.g. "chr2"
        - loop_start, loop_end  (bp)
        - region_start, region_end (bp)  <-- NEW
    """
    # ─── optional cropping for corigami loops ──────────────────────────────
    if corigami_loops:
        output = output[..., 52:-53, 52:-53] if output.ndim == 3 else output[52:-53, 52:-53]
        element["matrix"] = element["matrix"][52:-53, 52:-53]
        element["relative_loop_start"] -= 52
        element["relative_loop_end"]   -= 52

    # ─── HEADLINE (kb style) ───────────────────────────────────────────────
    headline = (
        f"{element['chr']}; Loop: "
        f"{_format_kb(element['loop_start'])} – {_format_kb(element['loop_end'])}"
    )

    # ─── FIGURE & GRID LAYOUT ──────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(nrows=2, ncols=2, height_ratios=[4, 1], hspace=0.25, wspace=0.25)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_exp  = fig.add_subplot(gs[0, 1])
    ax_gene = fig.add_subplot(gs[1, :])          # spans both columns

    # ─── MATRIX PLOTS (unchanged) ──────────────────────────────────────────
    global_min = torch.min(torch.min(output), torch.min(element["matrix"]))
    global_max = torch.max(torch.max(output), torch.max(element["matrix"]))
    stripe = element["status_filtered"]
    for ax, matrix, title in (
        (ax_pred, output, "Predicted CTCF HiChIP Matrix"),
        (ax_exp,  element["matrix"], "Experimental CTCF HiChIP Matrix"),
    ):
        arr = matrix.detach().cpu().numpy() if isinstance(matrix, torch.Tensor) else matrix
        im  = ax.imshow(arr, cmap="Reds", vmin=global_min, vmax=global_max)

        rs, re = element["relative_loop_start"], element["relative_loop_end"] - 1

        if show_corner_values:
            for x, y in [(re, rs)]:
                val = arr[y, x]
                ax.text(
                    x, y - 0.5, f"{val:.1f}",
                    color="blue",
                    ha="center", va="bottom",
                    fontsize=9, zorder=4
                )
        else:
            ax.scatter([rs, re, re], [rs, rs, re], color="green", s=15, zorder=3)

        ax.set_title(title, pad=18)
        ratio = _stripe_ratio(matrix, element, stripe=stripe, ignore_k=ignore_k)
        ax.text(0.5, 1.00, f"Stripe ratio = {ratio:.3f}",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=10)
        
        if show_stripe_sums:
            x_stripe = arr[rs, rs + 15 : re].mean()
            y_stripe = arr[rs : re - 15, re-1].mean()

            ax.text(
                rs, rs, f"X STRIPE MEAN = {x_stripe:.1f}",
                va="bottom", fontsize=9, color="blue"
            )
            ax.text(
                re, re, f"Y STRIPE MEAN = {y_stripe:.1f}",
                va="bottom", fontsize=9, color="blue"
            )

        ax.set_xlabel("Bin")
        ax.set_ylabel("Bin")
        fig.colorbar(im, ax=ax, label="Interaction Count", shrink=0.7)

    # ------------------------------------------------------------
    #  GENE-TRACK PANEL  (replace your current block with this one)
    # ------------------------------------------------------------
    chrom        = element["chr"]
    region_start = element["region_start"]
    region_end   = element["region_end"]

    # 1. read & filter
    gene_df = (
        pd.read_csv(gene_csv)
        .query("chr == @chrom and tss >= @region_start and tss <= @region_end")
        .sort_values("start")           # important for greedy packing
        .reset_index(drop=True)
    )

    # 2. greedy lane assignment to avoid overlaps
    lanes   = []          # list of [lane_end_bp] tracking last occupied end
    lane_id = []          # lane index for each gene

    for _, row in gene_df.iterrows():
        placed = False
        for ln, last_end in enumerate(lanes):
            if row["start"] > last_end:   # fits in this lane
                lane_id.append(ln)
                lanes[ln] = row["end"]
                placed = True
                break
        if not placed:                    # need a new lane
            lane_id.append(len(lanes))
            lanes.append(row["end"])

    gene_df["lane"] = lane_id
    n_lanes         = len(lanes)

    # 3. drawing
    base_y      = 0.2
    lane_height = 0.35 if n_lanes <= 4 else 0.25   # squeeze if many lanes
    arrow_h     = 0.4 * (lane_height / 0.35)       # scale arrow height, too

    for _, row in gene_df.iterrows():
        y = base_y + row["lane"] * lane_height

        x_start_kb = max(row["start"],  region_start) / 1000
        x_end_kb   = min(row["end"],    region_end)   / 1000
        tss_kb     = row["tss"] / 1000

        # bar
        ax_gene.hlines(y=y, xmin=x_start_kb, xmax=x_end_kb,
                    color="black", linewidth=3)

        # direction triangle
        if row["strand"] == "+":
            tri = [(tss_kb, y),
                (tss_kb, y + arrow_h),
                (tss_kb + 0.3, y)]
        else:
            tri = [(tss_kb, y),
                (tss_kb, y + arrow_h),
                (tss_kb - 0.3, y)]
        ax_gene.add_patch(Polygon(tri, facecolor="black", edgecolor="black"))

        # label
        ax_gene.text(tss_kb, y - 0.15 * (lane_height / 0.35),
                    row["gene_name"],
                    ha="center", va="top", fontsize=8)

    # 4. cosmetics
    ax_gene.set_xlim(region_start / 1000, region_end / 1000)
    ax_gene.set_ylim(-0.5,
                    base_y + (n_lanes - 1) * lane_height + arrow_h + 0.4)
    ax_gene.set_xlabel("Genomic Position (Kb)")
    ax_gene.set_ylabel("Genes")
    ax_gene.set_yticks([])
    for side in ("top", "right", "left"):
        ax_gene.spines[side].set_visible(False)

    # ─── FINISH UP ─────────────────────────────────────────────────────────
    fig.suptitle(headline, fontsize=16, fontweight="bold", y=0.98)
    #fig.tight_layout()

    if save_png:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_modification(
    element: dict,
    pred,                           # ← original prediction
    modified,                       # ← matrix after editing
    index: int = 0,
    save_png: bool = False,
    corigami_loops: bool = False,
    gene_csv: str = "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/gencode_processed.csv",
    save_path: str | None = None
):
    """
    Plot the *modified* Hi-C matrix and the difference (modified − pred),
    plus a gene-track panel underneath.

    Required keys in `element` (same as before):
        - relative_loop_start, relative_loop_end
        - chr, loop_start, loop_end, region_start, region_end
    """

    # ─── optional cropping ────────────────────────────────────────────────
    if corigami_loops:
        slicer = (..., 52, -53) if pred.ndim == 3 else (52, -53)
        pred      = pred[..., 52:-53, 52:-53]  if pred.ndim == 3 else pred[52:-53, 52:-53]
        modified  = modified[..., 52:-53, 52:-53] if modified.ndim == 3 else modified[52:-53, 52:-53]
        element["relative_loop_start"] -= 52
        element["relative_loop_end"]   -= 52

    # ─── HEADLINE ─────────────────────────────────────────────────────────
    headline = (
        f"{element['chr']}; Loop: "
        f"{_format_kb(element['loop_start'])} – {_format_kb(element['loop_end'])}"
    )

    # ─── figure & layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(nrows=2, ncols=2, height_ratios=[4, 1], hspace=0.25, wspace=0.25)
    ax_mod  = fig.add_subplot(gs[0, 0])
    ax_diff = fig.add_subplot(gs[0, 1])
    ax_gene = fig.add_subplot(gs[1, :])          # spans both columns

    # ─── left panel: modified matrix ─────────────────────────────────────
    global_min = torch.min(torch.min(pred), torch.min(modified)).item()
    global_max = torch.max(torch.max(pred), torch.max(modified)).item()

    arr_mod = modified.detach().cpu().numpy() if isinstance(modified, torch.Tensor) else modified
    im_mod  = ax_mod.imshow(arr_mod, cmap="Reds", vmin=global_min, vmax=global_max)

    rs, re = element["relative_loop_start"], element["relative_loop_end"] - 1
    ax_mod.scatter([rs, re, re], [rs, rs, re], color="green", s=15, zorder=3)

    ax_mod.set_title("Modified CTCF HiChIP Matrix")
    ax_mod.set_xlabel("Bin")
    ax_mod.set_ylabel("Bin")
    fig.colorbar(im_mod, ax=ax_mod, label="Interaction Count", shrink=0.7)

    # ─── right panel: difference (modified − pred) ───────────────────────
    diff = (modified - pred).detach() if isinstance(modified, torch.Tensor) else modified - pred
    diff_np = diff.cpu().numpy() if isinstance(diff, torch.Tensor) else diff
    max_abs = abs(diff_np).max()
    im_diff = ax_diff.imshow(
        diff_np,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax= max_abs,
    )

    ax_diff.scatter([rs, re, re], [rs, rs, re], color="green", s=15, zorder=3)
    ax_diff.set_title("Difference (Modified − Predicted)")
    ax_diff.set_xlabel("Bin")
    ax_diff.set_ylabel("Bin")
    fig.colorbar(im_diff, ax=ax_diff, label="Δ Interaction Count", shrink=0.7)

    # ─── gene-track panel (unchanged) ────────────────────────────────────
    chrom        = element["chr"]
    region_start = element["region_start"]
    region_end   = element["region_end"]

    gene_df = (
        pd.read_csv(gene_csv)
          .query("chr == @chrom and tss >= @region_start and tss <= @region_end")
          .reset_index(drop=True)
    )

    base_y, lane_height = 0.25, 0.35
    for i, row in gene_df.iterrows():
        y = base_y + (i % 2) * lane_height
        x_start_kb = max(row["start"],  region_start) / 1000
        x_end_kb   = min(row["end"],    region_end)   / 1000
        tss_kb     = row["tss"] / 1000

        ax_gene.hlines(y=y, xmin=x_start_kb, xmax=x_end_kb, color="black", linewidth=3)

        if row["strand"] == "+":
            tri = [(tss_kb, y), (tss_kb, y + 0.4), (tss_kb + 0.3, y)]
        else:
            tri = [(tss_kb, y), (tss_kb, y + 0.4), (tss_kb - 0.3, y)]
        ax_gene.add_patch(Polygon(tri, facecolor="black", edgecolor="black"))

        ax_gene.text(tss_kb, y - 0.2, row["gene_name"], ha="center", va="top", fontsize=8)

    ax_gene.set_xlim(region_start / 1000, region_end / 1000)
    ax_gene.set_ylim(-0.5, base_y + lane_height + 0.6)
    ax_gene.set_xlabel("Genomic Position (Kb)")
    ax_gene.set_ylabel("Genes")
    ax_gene.set_yticks([])
    for side in ("top", "right", "left"):
        ax_gene.spines[side].set_visible(False)

    # ─── finish up ───────────────────────────────────────────────────────
    fig.suptitle(headline, fontsize=16, fontweight="bold", y=0.98)
    #fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_png:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_simple(element, output, epoch):
    """
    Plots the predicted and actual Hi-C matrices with loop positions.

    Args:
        element (dict): Contains matrix information and loop positions.
        output (torch.Tensor): Predicted output from the model.
    """
    global_min = torch.min(torch.min(output), torch.min(element))
    global_max = torch.max(torch.max(output), torch.max(element))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Define matrices, titles, and colorbars
    matrices = [output.detach().cpu().squeeze().numpy(), element.detach().cpu().squeeze().numpy()]

    titles = ['Predicted CTCF HiChIP Matrix', 'Experimental CTCF HiChIP Matrix']
    colorbars = ['Interaction Count', 'Interaction Count']

    # Plot each matrix
    for ax, matrix, title, colorbar_label in zip(axes, matrices, titles, colorbars):
        ax.imshow(matrix, cmap='coolwarm', vmin=global_min.item(), vmax=global_max.item())
        ax.set_title(title)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Bin")
        ax.figure.colorbar(ax.images[0], ax=ax, label=colorbar_label, shrink=0.7)

    # Show the plots
    #plt.tight_layout()

    folder = "/cluster/work/boeva/shoenig/ews-ml/prelim_results/corigami/borzoi-lora-noL2"
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"{epoch}.png")
    fig.savefig(save_path)
    plt.close(fig)


def load_bigwig_signal(bigwig_path: str,
                       chrom: str,
                       start: int,
                       end: int) -> np.ndarray:
    """
    Load signal values from a BigWig file for a given region.
    """
    bw = pyBigWig.open(bigwig_path)
    # fetch values; fill missing with zero
    values = bw.values(chrom, start, end, numpy=True)
    bw.close()
    return np.nan_to_num(values)


def compute_importance_scores(model: torch.nn.Module,
                              element: dict,
                              stripe: str = "X",
                              ignore_k: int = 0,
                              device: str = "cpu") -> np.ndarray:
    """
    Compute attribution scores (e.g., Integrated Gradients) for the input.
    """
    scorer = GradientScorer(device)
    igrad, _ = scorer.compute_scores(model, element, stripe, ignore_k)
    return igrad.detach().cpu().numpy()


def plot_logo(attrib_tensor: np.ndarray,
              seq_start: int,
              seq_end: int,
              ax: plt.Axes,
              title: str,
              highlight_idx: int | list[int] | None = None,
              ymin=None,
              ymax=None):
    """
    Render a sequence logo on a given Axes, highlighting intervals.
    """
    # build dataframe for the region
    df = pd.DataFrame(
        attrib_tensor[seq_start:seq_end, :4],
        columns=["A", "C", "G", "T"]
    )
    logo = logomaker.Logo(df, color_scheme="classic", ax=ax)

    mn, mx = float(attrib_tensor[seq_start:seq_end, :4].min()), float(attrib_tensor[seq_start:seq_end, :4].max())
    if ymin is not None:
        mn = ymin
        mx = ymax
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left"], visible=True, bounds=[mn, mx])
    ax.set_xticks([])
    ax.set_yticks([mn, mx])
    ax.set_yticklabels([f"{mn:.2f}", f"{mx:.2f}"])
    ax.set_ylim(mn, mx)
    ax.set_ylabel("Value magnitude")
    ax.set_title(title)

    if highlight_idx is not None:
        # normalise to list
        if isinstance(highlight_idx, int):
            highlight_idx = [highlight_idx]
        if ymin is None:
            ymin, ymax = ax.get_ylim()
        for i in highlight_idx:
            if 0 <= i < (seq_end - seq_start):            # ignore out-of-range
                rect = patches.Rectangle(
                    (i - 0.5, ymin),                      # (x, y) lower left
                    width=1.0,
                    height=ymax - ymin,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(rect)


def download_enformer_weights():
    """
    Downloads the pre-trained weights from Hugging Face.
    """
    print("Downloading pre-trained weights...")
    local_save_path = './pretrained_weights'
    snapshot_download(
        repo_id='EleutherAI/enformer-official-rough',
        local_dir=local_save_path,
        local_dir_use_symlinks=False
    )
    print("Weights downloaded to:", local_save_path)

def download_borzoi_weights():
    """
    Downloads the pre-trained weights from Hugging Face.
    """
    print("Downloading pre-trained weights...")
    local_save_path = './pretrained_weights'
    snapshot_download(
        repo_id='johahi/flashzoi-replicate-0',
        local_dir=local_save_path,
        local_dir_use_symlinks=False
    )
    print("Weights downloaded to:", local_save_path)


def plot_and_save_matrices(
    element: dict,
    pred_init: torch.Tensor,          # initial prediction (pre-mutation)
    pred_mut: torch.Tensor,           # prediction after mutation
    save_dir: str,
    prefix: str,
    *,
    corigami_loops: bool = False,
    show_loop_markers: bool = True,
    show_stripe_ratio: bool = False,  # ← behind a boolean, off by default
    stripe: str | None = None,        # if None, will try element.get("status_filtered","X")
    ignore_k: int = 15,
    dpi: int = 300,
    cmap: str = "Reds",
    diff_cmap: str = "RdBu_r",
    show: bool = False,               # set True to also display
    close_figures: bool = True,       # close to free memory after saving
):
    """
    Save four images:
      1) Experimental matrix (element['matrix'])
      2) Initial prediction (pred_init)
      3) Post-mutation prediction (pred_mut)
      4) Difference = pred_mut − pred_init

    Notes:
      • The first three share the same vmin/vmax.
      • The difference uses symmetric +/- max(|Δ|).
      • No gene track. Optional loop markers. Optional stripe-ratio text.
      • Supports optional 'corigami_loops' cropping exactly like older code.

    Required keys in `element`:
      - 'matrix' (N×N), 'relative_loop_start', 'relative_loop_end'
      - optionally 'chr', 'loop_start', 'loop_end' for titles (not required)
      - optionally 'status_filtered' if you want stripe auto-selection

    Returns:
      dict with file paths {'exp','pre','post','diff'}.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- helpers ----------------
    def _to_2d_numpy(x):
        # Accept torch.Tensor or np.ndarray; squeeze a singleton channel if present.
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            return x.numpy()
        elif isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            return x
        else:
            # element["matrix"] might already be numpy
            return np.asarray(x)

    def _maybe_crop(arr):
        # Apply the corigami crop convention used in your previous functions.
        if not corigami_loops:
            return arr
        if arr.ndim == 2:
            return arr[52:-53, 52:-53]
        elif arr.ndim == 3:
            return arr[..., 52:-53, 52:-53]
        else:
            raise ValueError("Unsupported array ndim for cropping.")

    def _prep_for_plot(x):
        return _to_2d_numpy(_maybe_crop(x))

    def _draw_loop(ax):
        rs = element["relative_loop_start"]
        re = element["relative_loop_end"] - 1  # make end inclusive
        if corigami_loops:
            rs -= 52
            re -= 52
        ax.scatter([rs, re, re], [rs, rs, re], color="green", s=15, zorder=3)

    def _stripe_ratio_text(ax, arr2d):
        nonlocal stripe
        if not show_stripe_ratio:
            return
        # fallback to element status if not provided
        stripe = stripe or element.get("status_filtered", "X")
        # compute with torch to reuse your existing utility semantics
        t = torch.from_numpy(arr2d)
        try:
            ratio = _stripe_ratio(t, element, stripe=stripe, ignore_k=ignore_k)
            ax.text(0.5, 1.00, f"Stripe ratio = {ratio:.3f}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        except Exception:
            # stay silent if anything is missing
            pass

    # ---------------- prepare matrices ----------------
    exp_np  = _prep_for_plot(element["matrix"])
    pre_np  = _prep_for_plot(pred_init)
    post_np = _prep_for_plot(pred_mut)

    # Shared scale for the first three
    vmin = float(np.min([exp_np.min(), pre_np.min(), post_np.min()]))
    vmax = float(np.max([exp_np.max(), pre_np.max(), post_np.max()]))

    # Difference (post - pre)
    diff_np = post_np - pre_np
    max_abs = float(np.abs(diff_np).max())

    # ---------------- titles & filenames ----------------
    chr_str   = element.get("chr", "")
    loop_str  = None
    if "loop_start" in element and "loop_end" in element:
        try:
            loop_str = f"{_format_kb(element['loop_start'])} – {_format_kb(element['loop_end'])}"
        except Exception:
            loop_str = f"{element['loop_start']} – {element['loop_end']}"
    head = f"{chr_str}; Loop: {loop_str}" if (chr_str and loop_str) else None

    f_exp  = os.path.join(save_dir, f"{prefix}_exp.png")
    f_pre  = os.path.join(save_dir, f"{prefix}_pred_pre.png")
    f_post = os.path.join(save_dir, f"{prefix}_pred_post.png")
    f_diff = os.path.join(save_dir, f"{prefix}_diff.png")

    # ---------------- plotting helper ----------------
    def _save_single(arr2d, title, out_path, *, vmin=None, vmax=None, cmap="Reds",
                     add_loop=True, add_ratio=False, cbar_label="Interaction Count"):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax)
        if add_loop and show_loop_markers:
            _draw_loop(ax)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Bin")
        ttl = title if head is None else f"{title}\n{head}"
        ax.set_title(ttl, pad=14)
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
        if add_ratio:
            _stripe_ratio_text(ax, arr2d)
        #fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        if close_figures:
            plt.close(fig)

    # ---------------- save four images ----------------
    _save_single(exp_np,  "Experimental CTCF HiChIP Matrix", f_exp,
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 add_loop=True, add_ratio=show_stripe_ratio)

    _save_single(pre_np,  "Predicted (Pre-mutation)",         f_pre,
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 add_loop=True, add_ratio=show_stripe_ratio)

    _save_single(post_np, "Predicted (Post-mutation)",        f_post,
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 add_loop=True, add_ratio=show_stripe_ratio)

    _save_single(diff_np, "Difference (Post − Pre)",          f_diff,
                 vmin=-max_abs, vmax= max_abs, cmap=diff_cmap,
                 add_loop=True, add_ratio=False, cbar_label="Δ Interaction Count")

    return {"exp": f_exp, "pre": f_pre, "post": f_post, "diff": f_diff}
