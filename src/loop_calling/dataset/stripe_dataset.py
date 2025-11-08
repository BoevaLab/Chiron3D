import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


class StripeDataset(Dataset):
    def __init__(self, csv_file_path, mode="train", val_chroms=None, test_chroms=None, more_info=False):
        """
        Args:
            csv_file_path (str): Path to the CSV file containing the processed loops.
            mode (str): Mode of the dataset ('train', 'val', 'test', or 'all').
            val_chroms (list): List of chromosomes to be used for validation.
            test_chroms (list): List of chromosomes to be used for testing.
            more_info (bool): If True, returns additional information.
        """
        self.csv_file_path = csv_file_path
        self.mode = mode
        self.val_chroms = val_chroms if val_chroms else []
        self.test_chroms = test_chroms if test_chroms else []
        self.more_info = more_info

        # Load and filter the regions based on the mode
        self.regions = self._read_regions_file()
        self.filtered_regions = self._filter_regions_by_mode()

    def _read_regions_file(self):
        """
        Reads the regions file and returns a list of dictionaries.
        """
        df = pd.read_csv(self.csv_file_path, sep='\t', converters={'stripe': literal_eval})
        return df.to_dict('records')

    def _filter_regions_by_mode(self):
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

    def __len__(self):
        return len(self.filtered_regions)

    def __getitem__(self, idx):
        region = self.filtered_regions[idx]
        stripe = region['stripe']
        stripe = torch.tensor(stripe, dtype=torch.float32)
        stripe = torch.log(stripe + 1)

        if self.more_info:
            tweed_status = region.get("enrich_status_Tweed", "N")
            sum_status = region.get("enrich_status_sum", "N")
            median_status = region.get("enrich_status_median", "N")
            filtered_status = region.get("status_filtered", "N")
            return stripe, {"tweed": tweed_status, "sum": sum_status, "median": median_status, "filtered": filtered_status}

        return stripe


