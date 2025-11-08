import numpy as np
import pandas as pd
from src.loop_calling.importance_analysis.importance_scoring import BaseImportanceScorer
from tqdm import tqdm


# --- Data Preparation Class to get data in right format for TF-MoDisco---
class DataPreparator:
    def __init__(self, dataset, model, device, blacklist_file_path, scorer: BaseImportanceScorer):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.blacklist_df = self._read_blacklist(blacklist_file_path)
        self.scorer = scorer
        self.filtered_statistics = {"blacklist": 0, "out_of_bounds": 0, "unknown_nucleotide": 0, "sparse": 0}
        self.OFFSET = 0
        self.RESOLUTION = 5_000
    
    def _stripe_zero_fraction(self, arr):
        return (arr == 0).sum() / arr.size if arr.size else np.nan

    def _mean_stripe_sparsity(self, element): 
        """
        Compute the mean stripe‑0‑fraction (vertical + horizontal)/2.
        """
        mat = element["matrix"]
        arr = (
            mat.detach().cpu().numpy()
            if hasattr(mat, "detach") else np.asarray(mat)
        )

        rs = int(element["relative_loop_start"])
        re = int(element["relative_loop_end"]) - 1  # inclusive

        stripe_vert  = arr[rs:re, rs]          # column rs
        stripe_horiz = arr[re, rs+1:re+1]      # row re

        v = self._stripe_zero_fraction(stripe_vert)
        h = self._stripe_zero_fraction(stripe_horiz)
        return 0.5 * (v + h)

    def _read_blacklist(self, blacklist_file_path):
        return pd.read_csv(blacklist_file_path, sep="\t", header=None, names=["chrom", "start", "end"])

    def check_filter_conditions(self, start_left, end_left, start_right, end_right, element):
        if start_left < 0 or end_right >= (element['region_end'] - element['region_start']):
            self.filtered_statistics["out_of_bounds"] += 1
            return False

        """if (element["sequence"][4, start_left:end_left].sum() != 0) or \
                (element["sequence"][4, start_right:end_right].sum() != 0):
            self.filtered_statistics["unknown_nucleotide"] += 1
            return False""" # TODO FIND SOLUTION FOR THIS THATS IS COMPATIBLE

        left_start = element["loop_start"] - self.RESOLUTION
        left_end = element["loop_start"]
        right_start = element["loop_end"] - self.RESOLUTION
        right_end = element["loop_end"]

        chrom = element["chr"]
        blacklist = self.blacklist_df[self.blacklist_df["chrom"] == chrom]

        # If *either* flank overlaps any blacklist interval, filter it out:
        for _, row in blacklist.iterrows():
            bstart = row["start"]
            bend = row["end"]

            # overlap occurs if interval starts before our end and ends after our start
            left_overlaps = (bstart < left_end) and (bend > left_start)
            right_overlaps = (bstart < right_end) and (bend > right_start)

            if left_overlaps or right_overlaps:
                self.filtered_statistics["blacklist"] += 1
                return False
            
        sparsity = self._mean_stripe_sparsity(element)
        if sparsity > 0.25:
            self.filtered_statistics["sparse"] += 1
            return False

        return True

    def prepare_data(self, stripe):
        sequences_left = []
        scores_left = []
        sequences_right = []
        scores_right = []
        convergence_deltas = []

        # Filter out loops that are not of the specific loop type ("X", "Y", "X,Y")
        for element in tqdm(self.dataset):
            if element["status_filtered"] != stripe:
                continue

            # Define region indices based on element properties
            start_left = element["loop_start"] - element["region_start"] - self.OFFSET
            end_left = element["loop_start"] - element["region_start"] + self.RESOLUTION + self.OFFSET

            start_right = element["loop_end"] - element["region_start"] - self.RESOLUTION - self.OFFSET
            end_right = element["loop_end"] - element["region_start"] + self.OFFSET

            if not self.check_filter_conditions(start_left, end_left, start_right, end_right, element):
                continue

            # Mapping for A, C, G, T
            #nucleotide_indices = [0, 2, 3, 1] TODO: FIND SOLUTION HERE TOO

            # Extract nucleotide sequences
            sequence_left = element["sequence"][:, start_left:end_left].clone()
            sequence_right = element["sequence"][:, start_right:end_right].clone()

            # Compute scores using the chosen scorer
            score_tensor, deltas = self.scorer.compute_scores(self.model, element, stripe, ignore_k=15)

            if score_tensor is None:
                continue

            score_left = score_tensor[start_left:end_left, :].clone()
            score_right = score_tensor[start_right:end_right, :].clone()

            sequences_left.append(sequence_left.cpu().numpy())
            scores_left.append(score_left.detach().cpu().numpy())
            sequences_right.append(sequence_right.cpu().numpy())
            scores_right.append(score_right.detach().cpu().numpy())
        
        # Assure everything is in length-last format

        sequences_left  = ensure_length_last(np.array(sequences_left))
        scores_left     = ensure_length_last(np.array(scores_left))
        sequences_right = ensure_length_last(np.array(sequences_right))
        scores_right    = ensure_length_last(np.array(scores_right))

        print(f"Filtered statistics: {self.filtered_statistics}")
        return sequences_left, scores_left, sequences_right, scores_right
    


def ensure_length_last(x: np.ndarray) -> np.ndarray:
    """
    For a 3-D NumPy array with shape (batch, A, B), 
    make sure the longest of A and B is in the last axis.
    """
    if x.ndim == 3:
        a, b = x.shape[1], x.shape[2]
        if b < a:
            x = x.swapaxes(1, 2)
    x = x[:, :4, :]
    return x

