import argparse
import pandas as pd
import numpy as np
import cooler
from tqdm import tqdm


def get_args():
    FILE_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/processed/A673_WT/loops.csv"
    COOL_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool"#"/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/raw/HeLaS3/GSE108869_HeLaS3_CTCF_5kb.cool"
    OUTPUT_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/processed/A673_WT"

    parser = argparse.ArgumentParser(description="Loop Processing Pipeline")
    parser.add_argument('--file_path', dest='file_path', default=FILE_PATH, help="Path to the loop calling file")
    parser.add_argument('--cool_path', dest='cool_path', default=COOL_PATH, help="Path to the .cool file")
    parser.add_argument('--output_path', dest='output_path', default=OUTPUT_PATH, help="Output directory")

    return parser.parse_args()


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    return df


def is_strong_signal(arr, min_lower, lower_quantile, min_median):
    lower_quantile = np.quantile(arr, lower_quantile)
    median_value = np.median(arr)
    return (lower_quantile >= min_lower) and (median_value >= min_median)


def filter_sym(df, cool):
    """
    Function to filter symmetric loops
    """
    # Iterate over the rows of symmetric loops (directly in the original dataframe)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering loops"):
        # Create a list of the three status values
        statuses = [row['enrich_status_Tweed'], row['enrich_status_median'], row['enrich_status_sum']]
        if row['enrich_status_sum'] == 'X':
            df.at[idx, 'status_filtered'] = 'X'
        elif row['enrich_status_sum'] == 'Y':
            df.at[idx, 'status_filtered'] = 'Y'
        elif statuses.count("X,Y") == 3:
            chr_nr = row['chr'][3:]
            loop_start = row['loop_start']
            loop_end = row['loop_end']
            m = cool.matrix(balance=False).fetch(f"{chr_nr}:{loop_start}-{loop_end}")
            x_stripe = m[0, 15:]
            y_stripe = m[:-15, -1]
            min_lower = 2
            lower_quantile = 0.1
            min_median = 3
            if is_strong_signal(x_stripe, min_lower, lower_quantile, min_median) and\
                    is_strong_signal(y_stripe, min_lower, lower_quantile, min_median):
                df.at[idx, 'status_filtered'] = 'X,Y'
    return df

def filter_asym(df, cool):
    """
    Function to filter asymmetric loops
    """
    # Iterate over the rows of symmetric loops (directly in the original dataframe)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering loops"):
        # Create a list of the three status values
        statuses = [row['enrich_status_Tweed'], row['enrich_status_median'], row['enrich_status_sum']]
        if statuses.count("X") >= 2:
            df.at[idx, 'status_filtered'] = 'X'
        elif statuses.count("Y") >= 2:
            df.at[idx, 'status_filtered'] = 'Y'
    return df


def main():
    args = get_args()

    # Load data
    df = load_data(args.file_path)

    df['status_filtered'] = 'N'  # Default not assigned to either X,Y or X or Y

    cool = cooler.Cooler(args.cool_path)
    df = filter_sym(df, cool)
    #df = filter_asym(df, cool)
    df.to_csv(f"{args.output_path}/filtered_loops.csv", sep="\t", index=False)


if __name__ == "__main__":
    main()
