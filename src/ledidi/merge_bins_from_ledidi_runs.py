import argparse
import ast
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
from src.loop_calling.dataset.loop_dataset import LoopDataset


BIN_TEST_CHROMS = {
    1: ["chr1"],
    2: ["chr2", "chr8"],
    3: ["chr3", "chr12", "chr17", "chr22"],
    4: ["chr4", "chr7", "chr10", "chr15"],
    5: ["chr5", "chr6", "chr11", "chr16"],
    6: ["chr13", "chr20", "chr19"],
    7: ["chr9", "chr14", "chr18", "chr21"],
}

BASE_TRAIN_RUN_DIR = Path(
    "/cluster/work/boeva/shoenig/ews-ml/training_runs_ledidi_A673_WT"
)
OUTPUT_ROOT_DIR = Path("/cluster/work/boeva/shoenig/ews-ml/output_matrices_ledidi")
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)  # ensure the dir exists


def build_dataset(bin_id: int) -> LoopDataset:
    return LoopDataset(
        regions_file_path=(
            "/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/"
            "A673_WT/500kb_loops.csv"
        ),
        cool_file_path=(
            "/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/"
            "A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool"
        ),
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=BIN_TEST_CHROMS[bin_id],
        motif="",
        use_pretrained_backbone=True,
    )


def augment_bin(bin_id: int) -> pd.DataFrame:
    """
    Return a DataFrame that contains **all** columns from bin{bin_id}/ledidi_debug.csv
    plus three new columns:  bin_id, region_start, region_end.
    Only rows with exactly one edit (len(mutations_final) == 1) are kept.
    """
    bin_dir = BASE_TRAIN_RUN_DIR / f"bin{bin_id}"
    csv_path = bin_dir / "ledidi_debug.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # --- 1. load + keep only single-edit rows ------------------------------
    df = pd.read_csv(csv_path)
    df["mutations_final_parsed"] = df["mutations_final"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.strip() else []
    )
    df = df[df["mutations_final_parsed"].str.len() == 1].reset_index(drop=True)

    # --- 2. build the exact dataset for this bin ---------------------------
    dataset = build_dataset(bin_id)

    # --- 3. populate the new columns --------------------------------------
    df["bin_id"] = bin_id
    region_starts: List[int] = []
    region_ends: List[int] = []

    for idx in tqdm(df["idx"], desc=f"bin{bin_id}", leave=False):
        element = dataset[int(idx)]
        region_starts.append(element["region_start"])
        region_ends.append(element["region_end"])

    df["region_start"] = region_starts
    df["region_end"] = region_ends

    # we no longer need the helper column
    df.drop(columns="mutations_final_parsed", inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate ledidi_debug.csv tables from selected bins, "
            "adding bin_id, region_start and region_end."
        )
    )
    parser.add_argument(
        "--bins",
        nargs="*",
        type=int,
        default=list(range(1, 8)),
        help="Which bins to include (1-7). Default: all.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=OUTPUT_ROOT_DIR / "ledidi_debug_with_regions.csv",
        help="Destination CSV. Default: %(default)s",
    )
    args = parser.parse_args()

    dfs: List[pd.DataFrame] = []
    for bin_id in args.bins:
        if bin_id not in BIN_TEST_CHROMS:
            print(f"[ERROR] Unknown bin {bin_id}; skipping.")
            continue
        dfs.append(augment_bin(bin_id))

    if not dfs:
        print("Nothing collected — check your --bins selection.")
        return

    out_df = pd.concat(dfs, ignore_index=True)
    out_df.to_csv(args.outfile, index=False)
    print(f"[OK] Wrote {len(out_df):,} rows → {args.outfile}")


if __name__ == "__main__":
    main()
