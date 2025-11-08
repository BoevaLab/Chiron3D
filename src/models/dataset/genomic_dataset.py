import torch
from torch.utils.data import Dataset
import pyfaidx
import cooler
import numpy as np
import pandas as pd
import os
from src.models.dataset.utils import get_feature, onehotencode_dna, get_matrix
from src.models.dataset.augments import shift_aug, reverse_complement, gaussian_noise


class GenomicDataset(Dataset):
    def __init__(self, regions_file_path, cool_file_path, fasta_dir, genomic_feature_path=None,
                 mode="train", val_chroms=None, test_chroms=None, encode_motif="", use_pretrained_backbone=False, use_aug=False):
        """
        Args:
            regions_file_path (str): Path to the .bed file with the genomic regions.
            cool_file_path (str): Path to the .cool file with HiChIP data.
            fasta_dir (str): Directory containing chromosome FASTA files.
            genomic_feature_path (str, optional): Path to the folder with .bw files with the genomic feature data. Defaults to None.
            mode (str): One of "train", "val", "test" to specify the dataset mode.
            val_chroms (list, optional): List of chromosomes for validation. Defaults to None.
            test_chroms (list, optional): List of chromosomes for testing. Defaults to None.
            encode_motif (str): Which motif to 1hot encode as input. Defaults to empty string (nothing gets encoded in that case).
            use_pretrained_backbone (bool): Whether to use Enformer/Borzoi for embeddings. Defaults to False.
        """
        self.regions_file_path = regions_file_path
        self.cool_file_path = cool_file_path
        self.fasta_dir = fasta_dir
        self.genomic_feature_path = genomic_feature_path
        self.mode = mode

        self.bw_files = self._get_bw_files()
        self.val_chroms = set(val_chroms or [])
        self.test_chroms = set(test_chroms or [])
        self.regions = self._read_regions_file()
        self.cool = self._initialize_cooler()
        self._chrom_set = set(self.cool.chromnames)
        self.filtered_regions = self._filter_regions_by_mode()

        self.encode_motif = encode_motif
        self.use_aug = use_aug
        self.use_pretrained_backbone = use_pretrained_backbone
        self.dna_channels = 4 if use_pretrained_backbone else 5

    def __len__(self):
        return len(self.filtered_regions)

    def _get_bw_files(self):
        if self.genomic_feature_path is None:
            return []
        bw_files = [self.genomic_feature_path + "/" + f for f in os.listdir(self.genomic_feature_path) if f.endswith('.bw')]
        return bw_files

    def _read_regions_file(self):
        """Reads the regions file and returns a list of regions."""
        df = pd.read_csv(self.regions_file_path, sep='\t', names=["chr", "region_start", "region_end"])

        return df.to_dict('records')

    def _initialize_cooler(self):
        """Initializes the COOL object."""
        return cooler.Cooler(self.cool_file_path)

    def _filter_regions_by_mode(self):
        """Filters regions based on the dataset mode."""
        if self.mode == "train":
            return [
                region for region in self.regions
                if region.get("chr") not in self.val_chroms and region.get("chr") not in self.test_chroms
            ]
        elif self.mode == "val":
            return [region for region in self.regions if region.get("chr") in self.val_chroms]
        elif self.mode == "test":
            return [region for region in self.regions if region.get("chr") in self.test_chroms]
        elif self.mode == "all":
            return self.regions
        else:
            raise ValueError("Invalid mode. Choose from 'all', 'train', 'val', or 'test'.")

    def _normalize_chrom(self, chrom):
        """
        Different cool files have different standards for naming. This
        return a chromosome name that definitely exists in chromnames of the cool file.
        """
        if chrom in self._chrom_set:
            return chrom
        # strip “chr” if present
        if chrom.startswith("chr") and chrom[3:] in self._chrom_set:
            return chrom[3:]
        raise KeyError(f"Chromosome '{chrom}' not found in cooler file. "
                       f"Available: {list(self._chrom_set)[:5]}…")

    def _compute_motif_mask(self, sequence):
        sequence = sequence.lower()
        mask = np.zeros(len(sequence))
        motif_length = len(self.encode_motif)
        for i in range(len(sequence) - motif_length-1):
            if sequence[i:i + motif_length] == self.encode_motif.lower():
                mask[i] = 1  # Mark only the start position
        return mask[np.newaxis, :]  # Shape: (1, len)

    def build_output(self, chrom, output):
        fasta = pyfaidx.Fasta(f"{self.fasta_dir}/{output['chr']}.fa")
        sequence = onehotencode_dna(fasta[output['chr']][output['region_start']:output['region_end']].seq,
                                    self.dna_channels)
        matrix = get_matrix(self.cool, chrom, output['region_start'], output['region_end'])
        features = []
        for file_path in self.bw_files:
            features.append(get_feature(file_path, output['chr'], output['region_start'], output['region_end']))
        features_tensor = torch.cat(features, dim=0) if features else None

        if self.use_aug:
            sequence, features_tensor, matrix = reverse_complement(sequence, features_tensor, matrix, chance=0.5)

        if features_tensor is not None:
            output["features"] = features_tensor
        output["matrix"] = matrix
        output["sequence"] = sequence

        return output

    def __getitem__(self, idx):
        output = self.filtered_regions[idx].copy()
        output["region_start"] = int(output["region_start"])
        output["region_end"] = int(output["region_end"])
        chrom = self._normalize_chrom(output['chr'])
        chrom_len = self.cool.chromsizes[chrom]

        if self.mode == "train":
            output["region_start"], output["region_end"] = shift_aug(chrom_len, output["region_start"], output["region_end"])

        output = self.build_output(chrom, output)
        return output
