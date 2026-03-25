from src.models.evaluation.metrics import mse, insulation_corr
import torch
import numpy as np
from tqdm import tqdm
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])


def run_test(model, test_dataloader):
    pearson_list = []
    spearman_list = []
    mse_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print("DEBUG: Starting evaluation")

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        test_input = batch["sequence"]
        test_input = test_input.to(device)

        if "features" in batch:
            genom_feat = batch["features"].to(device)
            test_input = torch.cat((test_input, genom_feat), dim=1)

        with torch.no_grad():
            output = model(test_input)  # Get predicted output
            output = torch.clamp(output, min=0)
            # Loop through batch samples
            for out, true in zip(output, batch["matrix"]):
                out = out.cpu()  # Move output to CPU
                true = true.cpu()  # Move true matrix to CPU

                r_pearson, r_spearman = insulation_corr(out, true)
                mse_loss = mse(out, true)

                if not np.isnan(r_pearson):
                    pearson_list.append(r_pearson)
                if not np.isnan(r_spearman):
                    spearman_list.append(r_spearman)
                if not np.isnan(mse_loss):
                    mse_list.append(mse_loss)

    # Compute and print average Pearson correlation and MSE loss
    print("==========================================================================================")
    print(f"Average Pearson Correlation: {np.mean(pearson_list)}")
    print(f"Average Spearman Correlation: {np.mean(spearman_list)}")
    print(f"Average MSE loss: {np.mean(mse_list)}")

    return {"test_pearson": np.mean(pearson_list), "test_spearman": np.mean(spearman_list), "test_mse": np.mean(mse_list)}
