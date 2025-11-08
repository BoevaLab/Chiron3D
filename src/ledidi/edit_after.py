import argparse
import ast
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm
from src.loop_calling.dataset.loop_dataset import LoopDataset


def build_dataset() -> LoopDataset:
    return LoopDataset(
        regions_file_path="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv",
        cool_file_path="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool",
        fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes",
        genomic_feature_path=None,
        mode="test",
        val_chroms=[],
        test_chroms=["chr2"],
        use_pretrained_backbone=True
    )


dataset = build_dataset()

df = pd.read_csv("/cluster/work/boeva/shoenig/ews-ml/ledidi_tests/asym_to_sym/ledidi_debug_2.csv")

region_starts: List[int] = []
region_ends: List[int] = []

for idx in tqdm(df["idx"]):
    element = dataset[int(idx)]
    region_starts.append(element["region_start"])
    region_ends.append(element["region_end"])

df["region_start"] = region_starts
df["region_end"] = region_ends

df.to_csv("/cluster/work/boeva/shoenig/ews-ml/ledidi_tests/asym_to_sym/ledidi_debug_2_with_regions.csv")
