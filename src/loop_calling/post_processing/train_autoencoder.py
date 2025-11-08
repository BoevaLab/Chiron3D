from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from collections import defaultdict
from src.loop_calling.dataset.stripe_dataset import StripeDataset
from src.loop_calling.post_processing.autoencoder import AdaptiveConvAutoencoder


def main():
    LOOP_PATH = "/Volumes/scratch-boeva/data/projects/Sebastian_EWS/data/loop_calling/processed/HeLaS3/loops.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = StripeDataset(
        csv_file_path=LOOP_PATH,
        mode="train",
        val_chroms=["chr5", "chr12", "chr13", "chr21"],
        test_chroms=["chr2", "chr6", "chr19"])

    val_dataset = StripeDataset(
        csv_file_path=LOOP_PATH,
        mode="val",
        val_chroms=["chr5", "chr12", "chr13", "chr21"],
        test_chroms=["chr2", "chr6", "chr19"])

    print("Length of train_dataset:", len(train_dataset))
    print("Length of val_dataset:", len(val_dataset))

    def group_indices_by_difference(dataset):
        diff_indices = defaultdict(list)
        for i, row in enumerate(dataset):
            diff = int(len(row)//2)
            diff_indices[diff].append(i)
        return list(diff_indices.values())

    index_groups_train = group_indices_by_difference(train_dataset)
    index_groups_val = group_indices_by_difference(val_dataset)

    # Create DataLoaders for the index groups
    train_loader = DataLoader(index_groups_train, batch_size=1, shuffle=True)
    val_loader = DataLoader(index_groups_val, batch_size=1, shuffle=True)

    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AdaptiveConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    best_val_loss = float("inf")
    best_model_weights = None
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_samples = 0

        for index_group_train in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            data = torch.stack([train_dataset[i] for i in index_group_train]).to(device)
            data = data.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs[0], data)

            # Accumulate weighted loss
            train_loss += loss.item() * len(data)
            train_samples += len(data)

            loss.backward()
            optimizer.step()

        train_loss /= train_samples  # Average loss over total samples

        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for index_group_val in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                data = torch.stack([val_dataset[i] for i in index_group_val]).to(device)
                data = data.unsqueeze(1)

                outputs = model(data)

                loss = criterion(outputs[0], data)

                # Accumulate weighted loss
                val_loss += loss.item() * len(data)
                val_samples += len(data)

        val_loss /= val_samples
        # Print epoch losses
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            best_epoch = epoch

    torch.save(best_model_weights, f"/Users/sebastian/University/Master/mt/ews-ml/results/autoencoder/4dims.pth")
    print(f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")


if __name__ == "__main__":
    main()
