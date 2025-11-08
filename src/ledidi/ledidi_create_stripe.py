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
from src.ledidi.running_utils import col_to_base, plot_all_for_sequence, make_non_anchor_bin_mask
from src.ledidi.wrappers import scalar_from_wrapper, StripeWrapper
from src.ledidi.losses import stripe_diff_loss


def evaluate_element(elem, core_model, device, stripe, run_dir):
    """Run LEDIDI on one loop and return a dict of metrics."""

    original_elem = copy.deepcopy(elem)
    original_sequence = elem["sequence"].to(device)

    X = elem["sequence"].unsqueeze(0).to(device)
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]
    ignore_k=15

    with torch.no_grad():
        m0     = core_model(X)
        sum_x0 = m0[:, i,     i + ignore_k : j  ].sum(dim=-1).item()
        sum_y0 = m0[:, i : j - ignore_k, j - 1].sum(dim=-1).item()
    
    wrapper = StripeWrapper(core_model, i, j, stripe=stripe, base_sum_x=sum_x0, base_sum_y=sum_y0).to(device)

    with torch.no_grad():
        if stripe == "X": # Doing this because I want a dynamic threshold. This will not penalize if the sum of the non dominant stripe is within 20% of the dominant ones
            dynamic_thresh = 0.3 * sum_x0
        else:
            dynamic_thresh = 0.3 * sum_y0

        y_hat = wrapper(X)                      
        diff = y_hat.item()

        print(f"Threshold: {dynamic_thresh}")
        init_loss = stripe_diff_loss(y_hat, thresh=dynamic_thresh)

    if abs(diff) < dynamic_thresh: 
        os.makedirs(os.path.join(run_dir, elem["chr"]), exist_ok=True)         
        print(f"✓ Loop idx {elem.get('idx', -1)} already inside target band "
              f"({diff:.3f}); skipping optimisation.")
        return {
            "idx": int(elem.get("idx", -1)),
            "loop_start": elem.get("loop_start", None),
            "loop_end": elem.get("loop_end", None),
            "stripe": stripe,
            "original_diff": diff,
            "edited_diff":  diff,
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
    )

    input_mask = make_non_anchor_bin_mask(elem, stripe, device=device)

    X_bar = ledidi(
        model=wrapper,
        X=X,
        y_bar=torch.zeros((1, 1), device=device), 
        output_loss=lambda y, _: stripe_diff_loss(y, thresh=dynamic_thresh),
        l=0.08,
        tau=1.0,
        max_iter=3000,
        early_stopping_iter=1200,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=0.3,
        input_mask=input_mask,
        return_history=False
    )

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

    # ---------------------- pruning -------------------------
    updated_sequence_first = X_bar.squeeze()
    diff_mask_first = original_sequence != updated_sequence_first           

    cols_with_any_diff_first = diff_mask_first.any(dim=0)    
    diff_positions_first = torch.nonzero(cols_with_any_diff_first).flatten()
    diff_positions_first = diff_positions_first.tolist()
    print("Originall diff pos:")
    print(diff_positions_first)
    if len(diff_positions_first) > 200:
        print(f"Something went wrong, there were {len(diff_positions_first)} proposed")

        return {
                "idx": int(elem.get("idx", -1)),
                "loop_start": elem.get("loop_start", None),
                "loop_end": elem.get("loop_end", None),
                "stripe": stripe,
                "original_diff": diff,
                "edited_diff":  diff,
                "mutations_raw": [],
                "mutations_final": [],
                "base_changes": [],
                "single_edit_contrib": None,
                "chr":   elem.get("chr", ""), 
                "contributions": [],
                "comment": "Weird loop, check it out"           
            }

    X_bar_p = greedy_pruning(wrapper, X, X_bar, threshold=5, verbose=True)

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
        y_hat_p = wrapper(X_bar_p).item()

    best_contrib = None
    contributions = []

    if len(diff_positions) > 1:
        single_diffs = []
        for pos in diff_positions:
            # Only doing one edit a time - and for quick visual exploration!
            one_edit_seq = original_sequence.clone()
            one_edit_seq[:, pos] = updated_sequence[:, pos]

            dif = scalar_from_wrapper(wrapper, one_edit_seq.to(device))
            single_diffs.append(dif)

        denom = max(1e-6, diff - y_hat_p)
        contributions = [(diff - d) / denom for d in single_diffs]
        print(f"Contributions: {contributions}")
        best_idx, best_contrib = max(enumerate(contributions), key=lambda x: x[1])

        if best_contrib >= 0.85:
            # keep only the winning edit
            keep_pos = diff_positions[best_idx]
            print(f"Keeping only edit at pos {keep_pos} (contribution = {best_contrib:.2%})")
        
            updated_sequence = original_sequence.clone()
            updated_sequence[:, keep_pos] = X_bar_p.squeeze()[:, keep_pos]
            diff_positions   = [keep_pos]
            y_hat_p          = single_diffs[best_idx]

            torch.save(updated_sequence.cpu(),
                   os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt"))
            
            mut_elem = copy.deepcopy(elem)
            mut_elem['sequence'] = updated_sequence.cpu()
            plot_all_for_sequence(
                mut_elem,
                updated_sequence,
                core_model,
                base_pred,
                idx_dir,
                tag="mutation_single",
                device=device
            )
        else:
            best_contrib = None
            tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
            torch.save(updated_sequence.cpu(), tensor_path)  
    else:
        best_contrib = None
        tensor_path = os.path.join(chr_dir, f"best_onehot_{elem['idx']}.pt")
        torch.save(updated_sequence.cpu(), tensor_path)  

    return {
        "idx": int(elem.get("idx", -1)),
        "loop_start": elem.get("loop_start", None),
        "loop_end": elem.get("loop_end", None),
        "stripe": stripe,
        "original_diff": diff,
        "edited_diff":  y_hat_p,
        "mutations_raw": copy.deepcopy(raw_positions),
        "mutations_final": copy.deepcopy(diff_positions),
        "base_changes": copy.deepcopy(raw_base_changes),
        "single_edit_contrib": best_contrib,
        "chr":   elem.get("chr", ""),    
        "contributions": contributions,
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
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=["chr6", "chr19"],
        use_pretrained_backbone=True
    )

    total_seen = 0
    results = []

    for idx in range(len(dataset)):
        elem = dataset[idx]
        stripe = elem.get("status_filtered", "").strip()
        if stripe not in {"X", "Y"}:
            continue  # skip non-qualifying loops
    

        print(f"Working with Index {idx}")
        total_seen += 1

        elem["idx"] = idx  # for reporting
        metrics = evaluate_element(
            elem,
            core_model,
            device,
            stripe,
            run_dir
        )
        results.append(metrics)

        if total_seen > 100:
            break

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, args.csv_out), index=False)

if __name__ == "__main__":
    main()
