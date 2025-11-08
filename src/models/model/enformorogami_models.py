import torch
import torch.nn as nn
import src.models.model.blocks as blocks
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import get_enformer_embeddings, freeze_all_but_last_n_layers_


def move_feature_forward(x):
    # Input: (B, L, C) -> Output: (B, C, L)
    return x.transpose(1, 2).contiguous()


def diagonalize_small(x):
    x_i = x.unsqueeze(2).repeat(1, 1, 56, 1)
    x_j = x.unsqueeze(3).repeat(1, 1, 1, 56)
    input_map = torch.cat([x_i, x_j], dim=1)
    return input_map


def get_enformer_model(local: bool):
    if local:
        return from_pretrained("EleutherAI/enformer-official-rough")
    return from_pretrained("/cluster/work/boeva/shoenig/enformer")


# Uses Enformer as backbone to obtain embeddings before diagonalization.
# 1D conv along channels and adaptive pooling along length dimension to bridge dims
class EnformerOrogami(nn.Module):

    def __init__(self, mid_hidden=128, local=False):
        super().__init__()

        self.enformer = get_enformer_model(local)

        # This utility function first freezes all layers, then unfreezes the last N transformer blocks.
        NUM_LAYERS_TO_FINETUNE = 1
        freeze_all_but_last_n_layers_(self.enformer, NUM_LAYERS_TO_FINETUNE)
        print(f"[DEBUG] Unfrozen the last {NUM_LAYERS_TO_FINETUNE} Transformer blocks of Enformer.")

        self.enformer.train()

        self.projector = nn.Sequential(
            nn.Conv1d(3072, 1024, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, mid_hidden, kernel_size=1, padding=0, bias=True)
        )
        self.length_reducer = nn.Sequential(
            # Stride 2 reduces length by half in each block
            # Block 1: 896 -> 448
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 2: 448 -> 224
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 3: 224 -> 112
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 4: 112 -> 56 (56 instead of 64 chans)
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU()
        )
        # self.projector = nn.Conv1d(3072, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.adaptive_enf_pool = nn.AdaptiveMaxPool1d(64)
        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((40, 40))

    def forward(self, x):
        x = move_feature_forward(x)
        x = get_enformer_embeddings(self.enformer, x, freeze=False, train_layernorms_only=False,
                                    train_last_n_layers_only=None)
        x = move_feature_forward(x)
        x = self.projector(x)
        x = self.length_reducer(x)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        x = torch.relu(x)
        x = self.adaptive_pool(x)
        return x


class EnformerOrogamiShallow(nn.Module):

    def __init__(self, mid_hidden=128, local=False):
        super().__init__()

        self.enformer = get_enformer_model(local)

        for param in self.enformer.parameters():
            param.requires_grad = False
        self.enformer.eval()

        self.projector = nn.Conv1d(3072, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.adaptive_enf_pool = nn.AdaptiveMaxPool1d(64)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((40, 40))

    def forward(self, x):
        x = move_feature_forward(x)
        x = get_enformer_embeddings(self.enformer, x, freeze=False)
        x = move_feature_forward(x)
        x = self.projector(x)
        x = self.adaptive_enf_pool(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        x = torch.relu(x)
        x = self.adaptive_pool(x)
        return x


class EnformerOrogamiDeep(nn.Module):

    def __init__(self, mid_hidden=128, local=False):
        super().__init__()

        self.enformer = get_enformer_model(local)

        for param in self.enformer.parameters():
            param.requires_grad = False
        self.enformer.eval()

        self.projector = nn.Sequential(
            nn.Conv1d(3072, 1024, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, mid_hidden, kernel_size=1, padding=0, bias=True)
        )
        self.length_reducer = nn.Sequential(
            # Stride 2 reduces length by half in each block
            # Block 1: 896 -> 448
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 2: 448 -> 224
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 3: 224 -> 112
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU(),
            # Block 4: 112 -> 56 (56 instead of 64 chans)
            nn.Conv1d(in_channels=mid_hidden, out_channels=mid_hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_hidden),
            nn.ReLU()
        )
        # self.projector = nn.Conv1d(3072, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.adaptive_enf_pool = nn.AdaptiveMaxPool1d(64)
        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((40, 40))

    def forward(self, x):
        x = move_feature_forward(x)
        x = get_enformer_embeddings(self.enformer, x, freeze=True)
        x = move_feature_forward(x)
        x = self.projector(x)
        # x = self.adaptive_enf_pool(x)
        x = self.length_reducer(x)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        x = torch.relu(x)
        x = self.adaptive_pool(x)
        return x
