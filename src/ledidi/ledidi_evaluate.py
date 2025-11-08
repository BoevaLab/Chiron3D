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
from src.ledidi.running_utils import col_to_base, plot_all_for_sequence
from src.ledidi.losses import ratio_ballpark_loss
from src.ledidi.wrappers import RatioWrapper, scalar_from_wrapper


def evaluate_element(elem, core_model, device, stripe, low, high, l_weight, max_iter, early_stop, tau, lr, run_dir):
    """Run LEDIDI on one loop and return a dict of metrics."""

    original_elem = copy.deepcopy(elem)
    original_sequence = elem["sequence"].to(device)

    X = elem["sequence"].unsqueeze(0).to(device)
    i, j = elem["relative_loop_start"], elem["relative_loop_end"]
    wrapper = RatioWrapper(core_model, i, j, stripe=stripe).to(device)

    with torch.no_grad():
        y_hat = wrapper(X)                      
        original_ratio = y_hat.item()
        init_loss = ratio_ballpark_loss(y_hat, low=low, high=high)

    if init_loss < 0.5: 
        os.makedirs(os.path.join(run_dir, elem["chr"]), exist_ok=True)         
        print(f"✓ Loop idx {elem.get('idx', -1)} already inside target band "
              f"({original_ratio:.3f}); skipping optimisation.")
        return {
            "idx": int(elem.get("idx", -1)),
            "loop_start": elem.get("loop_start", None),
            "loop_end": elem.get("loop_end", None),
            "stripe": stripe,
            "original_ratio": original_ratio,
            "edited_ratio":  original_ratio,
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

    X_bar = ledidi(
        model=wrapper,
        X=X,
        y_bar=torch.zeros((1, 1), device=device), 
        output_loss=lambda y, _: ratio_ballpark_loss(y, low=low, high=high),
        l=l_weight,
        tau=tau,
        max_iter=max_iter,
        early_stopping_iter=early_stop,
        device=device,
        batch_size=1,
        report_iter=100,
        lr=lr,
        return_history=False
    )

    # ---------------------- pruning -------------------------
    initial_diff_positions = torch.nonzero((original_sequence != X_bar.squeeze()).any(dim=0)   ).flatten().tolist()        
    print("Originall diff pos:")
    print(initial_diff_positions)
    if len(initial_diff_positions) > 200:
        print(f"Something went wrong, there were {len(initial_diff_positions)} proposed")

        return {
                "idx": int(elem.get("idx", -1)),
                "loop_start": elem.get("loop_start", None),
                "loop_end": elem.get("loop_end", None),
                "stripe": stripe,
                "original_ratio": original_ratio,
                "edited_ratio":  original_ratio,
                "mutations_raw": [],
                "mutations_final": [],
                "base_changes": [],
                "single_edit_contrib": None,
                "chr":   elem.get("chr", ""), 
                "contributions": [],
                "comment": "Weird loop, check it out"           
            }

    X_bar_p = greedy_pruning(wrapper, X, X_bar, threshold=0.2, verbose=True)

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
        single_ratios = []
        for pos in diff_positions:
            # Only doing one edit a time - and for quick visual exploration!
            one_edit_seq = original_sequence.clone()
            one_edit_seq[:, pos] = updated_sequence[:, pos]

            r = scalar_from_wrapper(wrapper, one_edit_seq.to(device))
            single_ratios.append(r)

        denom = max(1e-6, original_ratio - y_hat_p)
        contributions = [(original_ratio - r) / denom for r in single_ratios]
        print(f"Contributions: {contributions}")
        best_idx, best_contrib = max(enumerate(contributions), key=lambda x: x[1])

        if best_contrib >= 0.85:
            # keep only the winning edit
            keep_pos = diff_positions[best_idx]
            print(f"Keeping only edit at pos {keep_pos} (contribution = {best_contrib:.2%})")
        
            updated_sequence = original_sequence.clone()
            updated_sequence[:, keep_pos] = X_bar_p.squeeze()[:, keep_pos]
            diff_positions   = [keep_pos]
            y_hat_p          = single_ratios[best_idx]

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
        "original_ratio": original_ratio,
        "edited_ratio":  y_hat_p,
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
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l_weight", type=float, default=2.0, help="Weight on edit penalty (λ)")
    parser.add_argument("--low", type=float, default=0.5, help="Lower bound of acceptable ratio")
    parser.add_argument("--high", type=float, default=1.7, help="Upper bound of acceptable ratio")
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--early_stop", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=None, help="Process only first N qualifying loops (debug)")
    parser.add_argument("--run_dir", default=None, help="Where to store plots & logs (default: ./runs/<timestamp>/)")
    parser.add_argument("--bin", type=int, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "cmd.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading model")
    core_model = TrainModule.load_from_checkpoint(args.ckpt).to(args.device).eval()

    print("Loading dataset")
    dataset = load_dataset(args)

    results = []

    for idx in range(len(dataset)):
        elem = dataset[idx]
        stripe = elem.get("status_filtered", "").strip()
        if stripe not in {"X", "Y"}:
            continue  # skip non-qualifying loops
                
        print(f"Working with Index {idx}")
        elem["idx"] = idx
        metrics = evaluate_element(
            elem,
            core_model,
            args.device,
            stripe,
            args.low,
            args.high,
            args.l_weight,
            args.max_iter,
            args.early_stop,
            args.tau,
            args.lr,
            run_dir
        )
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, args.csv_out), index=False)


def load_dataset(args):
    if args.bin == 1:
        test_chroms = ["chr1"]
    elif args.bin == 2:
        test_chroms = ["chr2", "chr8"]
    elif args.bin == 3:
        test_chroms = ["chr3", "chr12", "chr17", "chr22"]
    elif args.bin == 4:
        test_chroms = ["chr4", "chr7", "chr10", "chr15"]
    elif args.bin == 5:
        test_chroms = ["chr5", "chr6", "chr11", "chr16"]
    elif args.bin == 6:
        test_chroms = ["chr13", "chr20", "chr19"]
    elif args.bin == 7:
        test_chroms = ["chr9", "chr14", "chr18", "chr21"]

    dataset = LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/TC71_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/TC71_WT/contact_matrix_data/TC71_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=test_chroms,
        motif="",
        use_pretrained_backbone=True
    )
    return dataset


if __name__ == "__main__":
    main()
   
