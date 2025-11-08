import argparse
import numpy as np
import torch
from src.loop_calling.importance_analysis.importance_scoring import GradientScorer, DeepLiftScorer, IntegratedGradientsScorer
from src.loop_calling.dataset.loop_dataset import LoopDataset
from src.loop_calling.importance_analysis.create_data import DataPreparator
from src.utils import load_model
from src.models.training.train import TrainModule
from src.models.model.corigami_models import ConvTransModelSmall
torch.serialization.add_safe_globals([argparse.Namespace])


def main():
    parser = argparse.ArgumentParser(description="Prepare tfmodisco data")
    parser.add_argument("--scoring", dest='scoring', choices=["gradient", "deeplift"], required=True)
    parser.add_argument("--stripe", dest='stripe', choices=["X", "Y", "X,Y", "STABLE"], required=True)
    parser.add_argument("--blacklist-file", dest='blacklist_file', type=str, required=True)
    parser.add_argument("--weights-path", dest='weights_path', type=str, required=True)

    parser.add_argument('--windows-file', dest='windows_file', help='Regions for training, validation and test data',
                        required=True)
    parser.add_argument('--fasta-dir', dest='fasta_dir', required=True, help='Directory with chromosome fasta files')
    parser.add_argument('--cool-file', dest='cool_file', required=True, help='Interaction matrix path')
    parser.add_argument('--genom-feat-path', dest='genom_feat_path', default=None,
                        help='Path to the genomic feature file (default: None)')
    parser.add_argument('--save-name', dest='save_name', required=True)
    parser.add_argument('--start-index', type=int, default=0, help='Start index for dataset chunk')
    parser.add_argument('--end-index', type=int, default=None, help='End index for dataset chunk')
    parser.add_argument('--motif', dest='motif', default="", help='Motif to encode (default: "")')
    parser.add_argument('--num-genom-feats', type=int, dest='num_genom_feats', default=1, help='Number of features')
    parser.add_argument('--enformer', action='store_true', help='Use enformer backbone for embeddings')
    parser.add_argument('--borzoi', action='store_true', help='Use enformer backbone for embeddings')

    args = parser.parse_args()

    use_pretrained_backbone = False
    if args.enformer or args.borzoi:
        use_pretrained_backbone = True

    if args.num_genom_feats == 0:
            args.genom_feat_path = None

    dataset = LoopDataset(
            regions_file_path=args.windows_file,
            cool_file_path=args.cool_file,
            fasta_dir=args.fasta_dir,
            genomic_feature_path=args.genom_feat_path,
            mode="all",
            val_chroms=["chr5", "chr12", "chr13", "chr21"],
            test_chroms=["chr2", "chr6", "chr19"],
            motif=args.motif,
            use_pretrained_backbone=use_pretrained_backbone
        )

    if args.end_index is not None:  # Slicing because of memory issues for the big symmetric dataset
        dataset = [dataset[i] for i in range(args.start_index, args.end_index)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainModule.load_from_checkpoint(args.weights_path, map_location=device)
    model.eval()

    if args.scoring == "gradient":
        scorer = GradientScorer(device)
    elif args.scoring == "integrated":
        scorer = IntegratedGradientsScorer(device)
    else:
        raise ValueError("Invalid scoring method selected.")

    preparator = DataPreparator(dataset, model, device, args.blacklist_file, scorer)

    #sequences_left, scores_left, sequences_right, scores_right, deltas = preparator.prepare_data(stripe=args.stripe)    
    sequences_left, scores_left, sequences_right, scores_right = preparator.prepare_data(stripe=args.stripe)    
    save_name = f"{args.save_name}_Stripe_{args.stripe}_Method_{args.scoring}"

    np.save(f"{save_name}_sequences_left.npy", sequences_left)
    np.save(f"{save_name}_scores_left.npy", scores_left)
    np.save(f"{save_name}_sequences_right.npy", sequences_right)
    np.save(f"{save_name}_scores_right.npy", scores_right)
    #np.save(f"{save_name}_convergence_deltas.npy", deltas)


if __name__ == "__main__":
    main()
