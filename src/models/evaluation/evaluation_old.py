from src.models.dataset.genomic_dataset import GenomicDataset
from src.models.model.corigami_models import ConvTransModelSmall
from src.models.model.enformorogami_models import EnformerOrogamiDeep
from src.utils import load_model, print_element
from src.models.evaluation.metrics import mse, insulation_corr
from src.models.training.module import TrainModule
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])


def init_parser():
    parser = argparse.ArgumentParser(description="C.Origami Evaluation Module")

    # Data file paths
    parser.add_argument('--regions-file', dest='regions_file', required=True,
                        help='Path to the genomic regions file (BED format)')
    parser.add_argument('--cool-file', dest='cool_file', required=True,
                        help='Path to the contact matrix file (COOL format)')
    parser.add_argument('--fasta-dir', dest='fasta_dir', required=True,
                        help='Directory containing the fasta/chromosome files')
    parser.add_argument('--genomic-feature', dest='genomic_feature_path', required=True,
                        help='Path to the genomic feature folder (consisting of bw file)')

    parser.add_argument('--motif', dest='motif', type=str, help='Include motif mask as additional genomic feature')

    # Model parameters
    parser.add_argument('--num-genom-feat', dest='num_genom_feat', type=int, default=0,
                        help='Number of genomic features to consider (default: 0)')

    # Model checkpoint path
    parser.add_argument('--ckpt-path', dest='ckpt_path', required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--borzoi', action='store_true', help='Use enformer backbone for embeddings')
    parser.add_argument('--clip105', action='store_true', help='Takes top left of Corigami prediction for fairer evaluation compared to Enformer')


    return parser.parse_args()


def main():
    args = init_parser()

    if args.num_genom_feat == 0:
            args.genomic_feature_path = None

    use_pretrained_backbone = False
    if args.borzoi:
        use_pretrained_backbone = True

    test_dataset = GenomicDataset(
        regions_file_path=args.regions_file,
        cool_file_path=args.cool_file,
        fasta_dir=args.fasta_dir,
        genomic_feature_path=args.genomic_feature_path,
        mode="test",
        val_chroms=["chr5", "chr12", "chr13", "chr21"],
        test_chroms=["chr2", "chr6", "chr19"],
        encode_motif=args.motif,
        use_pretrained_backbone=use_pretrained_backbone
    )

    print("Length test_dataset:", len(test_dataset))
    element = test_dataset[0]
    print("First element:")
    print_element(element)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = TrainModule.load_from_checkpoint(args.ckpt_path, map_location=device).to(device)
    model.eval()

    pearson_list = []
    spearman_list = []
    mse_list = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        test_input = batch["sequence"]
        test_input = test_input.to(device)
        if "features" in batch:
            genom_feat = batch["features"].to(device)
            test_input = torch.cat((test_input, genom_feat), dim=1)
        # Perform prediction
        with torch.no_grad():
            output = model(test_input)  # Get predicted output
            output = torch.clamp(output, min=0)  # Ensure non-negative values
            # Loop through batch samples
            for out, true in zip(output, batch["matrix"]):
                out = out.cpu()  # Move output to CPU
                true = true.cpu()  # Move true matrix to CPU
                if args.clip105:
                    true = true[:105, :105]
                    out = out[:105, :105]

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
    

if __name__ == '__main__':
    main()
