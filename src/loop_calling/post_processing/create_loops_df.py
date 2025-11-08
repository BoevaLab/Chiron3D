"""Asymmetric/Symmetric Loop Preparation
In order to use the **asymmetric** (with prominent x or y stripes only) or **symmetric** loops (with balanced stripes) that were identified by Tweed for interpretability measures using the adaptation of the C.Origami model, the loops must be contained inside regions of exactly 1,048,576 bases. This notebook does the following:
1. Load all the loops identified by Tweed.
2. Extracts the **asymmetric** or **symmetric** loops that are shorter than 1,048,576 bases.
3. Centers those loop regions within regions exactly 1,048,576 bases in length. The start positions of these regions are aligned to positions divisible by the resolution size. This is important as it ensures that the corresponding extracted contact matrices always have the same shape.
4. Saves the resulting DataFrame as a tab-separated file, which can then be used for dataset creation."""

import argparse
import pandas as pd
import numpy as np
import cooler
from tqdm import tqdm


def get_args():
    FILE_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/processed/A673_WT/A673_WT_CTCF_merged_tads_best_20201123_bis_true_20190526_i0_r5000_d15_e10_m2_fc2_filter.juicebox"
    COOL_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool"
    HG19_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/raw/HeLaS3/hg19.sizes"
    OUTPUT_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/processed/A673_WT"

    parser = argparse.ArgumentParser(description="Loop Processing Pipeline")
    parser.add_argument('--file_path', dest='file_path', default=FILE_PATH, help="Path to the loop calling file")
    parser.add_argument('--cool_path', dest='cool_path', default=COOL_PATH, help="Path to the .cool file")
    parser.add_argument('--hg19_path', dest='hg19_path', default=HG19_PATH, help="Path to the hg19 sizes file")
    parser.add_argument('--output_path', dest='output_path', default=OUTPUT_PATH, help="Output directory")

    return parser.parse_args()


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

    # Checking for mismatches in data
    mismatch = df[(df['chr1'] != df['chr2']) | (df['x1'] != df['y1']) | (df['x2'] != df['y2'])]
    if mismatch.empty:
        print("All rows have matching coordinates.")
    else:
        print(f"There are {len(mismatch)} mismatches in the following rows:")
        print(mismatch)

    # Rename columns and extract necessary information
    df_filter = df[['chr1', 'x1', 'x2']].copy()
    df_filter.rename(columns={'chr1': 'chr'}, inplace=True)

    df_filter['x1'] = df_filter['x1'].astype(int)
    df_filter['x2'] = df_filter['x2'].astype(int)

    df_filter['enrichX'] = df['score'].str.extract(r'enrichX=([\d.]+)').astype(float)
    df_filter['enrichY'] = df['score'].str.extract(r'enrichY=([\d.]+)').astype(float)
    df_filter['enrich_status'] = df['score'].str.extract(r'enrich_status=([\w,]+)')

    return df_filter


def add_signal(df, cool, signal_type='sum'):
    first_row_values = []
    last_column_values = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Calculating {signal_type}s"):
        chr_name = row['chr']
        x1 = row['x1']
        x2 = row['x2']

        matrix = cool.matrix(balance=False).fetch(f"{chr_name[3:]}:{x1}-{x2}")

        x_stripe = matrix[0, 15:]  # Exclude the first 15 bins near the diagonal
        y_stripe = matrix[:-15, -1]

        if signal_type == 'sum':
            first_row_value = x_stripe.sum()  # Ignore the first 15 bins closest to the diagonal
            last_column_value = y_stripe.sum()
        elif signal_type == 'median':
            first_row_value = np.median(x_stripe)
            last_column_value = np.median(y_stripe)

        first_row_values.append(first_row_value)
        last_column_values.append(last_column_value)

    df = calculate_enrichment(df, signal_type, first_row_values, last_column_values)

    return df


def add_stripe(df, cool):
    stripe_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding stripes"):
        chr_name = row['chr']
        x1 = row['x1']
        x2 = row['x2']

        matrix = cool.matrix(balance=False).fetch(f"{chr_name[3:]}:{x1}-{x2}")

        x_stripe = matrix[0, 15:]  # Exclude the first 15 bins near the diagonal
        y_stripe = matrix[:-15, -1]
        combined_stripe = np.concatenate([x_stripe, y_stripe])
        stripe_data.append(combined_stripe.tolist())

    df['stripe'] = stripe_data

    return df


def calculate_enrichment(df, operation, first_row_values, last_column_values):
    colX = f'enrichX_{operation}'
    colY = f'enrichY_{operation}'
    scoreCol = f'score_{operation}'
    statusCol = f'enrich_status_{operation}'

    df[colX] = first_row_values
    df[colY] = last_column_values

    if operation == 'median':
        df[scoreCol] = np.log2(df[colX] / (df[colY] + 1e-5) + 1e-5)
    else:
        df[scoreCol] = np.log2(df[colX] / df[colY])

    score_mean = df[scoreCol].mean()
    score_std_dev = df[scoreCol].std()

    df[statusCol] = df[scoreCol].apply(
        lambda score: 'X' if score >= score_mean + 2 * score_std_dev
        else 'Y' if score <= score_mean - 2 * score_std_dev
        else "X,Y"
    )

    return df


def center_loop(df, resolution, region_size):
    df = df.copy()
    df['midpoint'] = (df['x1'] + df['x2']) / 2
    df['region_start'] = df['midpoint'] - region_size / 2
    df['region_end'] = df['midpoint'] + region_size / 2
    df['remainder'] = df['region_start'] % resolution
    df['region_start'] -= df['remainder']
    df['region_end'] -= df['remainder']

    return df


def filter_invalid_loops(df, region_size, resolution, hg19_path):
    # Filter out loops that are larger than the region size
    bigger_regions = df[(df['x2'] - df['x1']) >= (region_size - 2 * resolution)]
    df = df[(df['x2'] - df['x1']) < (region_size - 2 * resolution)]
    print(f"Filtered out {len(bigger_regions)} loops larger than the region size.")

    # Center the loops
    df = center_loop(df, resolution, region_size)

    # Load chromosome sizes
    chrom_sizes = pd.read_csv(hg19_path, sep='\t', header=None, index_col=0).iloc[:, 0].to_dict()

    # Check if the regions are valid (within chromosome boundaries)
    valid_regions = df.apply(lambda row:
                                    (row['region_start'] >= 0) and
                                    (row['region_end'] <= chrom_sizes.get(row['chr'], 0)), axis=1)

    # Filter out invalid regions
    df = df[valid_regions]

    print(f"Filtered out {len(df) - valid_regions.sum()} invalid regions.")

    return df


def main():
    args = get_args()

    # Load data
    df = load_data(args.file_path)

    # Load cooler file
    cool = cooler.Cooler(args.cool_path)

    # Add signals
    df = add_stripe(df, cool)
    df = add_signal(df, cool, signal_type='sum')
    df = add_signal(df, cool, signal_type='median')

    # Log the enrichment status
    print(f"Tweed enrichment status: {df['enrich_status'].value_counts()}")
    print(f"Median enrichment status: {df['enrich_status_median'].value_counts()}")
    print(f"Sum enrichment status: {df['enrich_status_sum'].value_counts()}")

    # Filter and center loops
    df = filter_invalid_loops(df, 1048576, 5000, args.hg19_path)
    # Save the results
    df = df.rename(columns={'x1': 'loop_start', 'x2': 'loop_end', 'enrichX': 'enrichX_Tweed', 'enrichY': 'enrichY_Tweed',
                   'enrich_status': 'enrich_status_Tweed'})
    df = df.drop(columns=['remainder', 'midpoint'])
    df.to_csv(f"{args.output_path}/loops.csv", sep='\t', index=False)


if __name__ == "__main__":
    main()


