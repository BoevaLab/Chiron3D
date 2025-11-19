import torch
import torch.nn as nn
import src.models.model.blocks as blocks
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from einops import rearrange

# TODO: Allow for borzoi paths

def diagonalize_small(x):
    x_i = x.unsqueeze(2).repeat(1, 1, 105, 1)
    x_j = x.unsqueeze(3).repeat(1, 1, 1, 105)
    input_map = torch.cat([x_i, x_j], dim=1)
    return input_map

def move_feature_forward(x):
    # Input: (B, L, C) -> Output: (B, C, L)
    return x.transpose(1, 2).contiguous()

def get_borzoi_backbone(local: bool, model_type: str):
    assert model_type in ["borzoi", "flashzoi"], "Invalid model type. Choose 'borzoi' or 'flashzoi'."
    cfg = BorzoiConfig.from_pretrained(f"../../data/{model_type}")
    cfg.return_center_bins_only = False # forces 16,352 bins
    borzoi = Borzoi.from_pretrained(f"../../data/{model_type}", config=cfg)
    return borzoi


class Chiron3D(nn.Module):

    def __init__(self, mid_hidden=128, local=False, model_type="borzoi"):
        super().__init__()

        self.borzoi = get_borzoi_backbone(local, model_type)

        for param in self.borzoi.parameters():
            param.requires_grad = False
        self.borzoi.eval()

        self.activation = nn.ReLU()
        self.projector = nn.Conv1d(1536, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)

        self.length_reducer = nn.AdaptiveAvgPool1d(105)

        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128)

    def forward(self, x):
        x = self.borzoi.get_embs_after_crop(x)
        x = self.projector(x)
        x = self.length_reducer(x)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        return x


class ResidualDownBlock(nn.Module):
    def __init__(self, ch, kernel_size, stride):
        super().__init__()
        # main conv path
        self.conv = nn.Conv1d(ch, ch, kernel_size, stride=stride, padding=0)
        self.bn = nn.GroupNorm(num_groups=1, num_channels=ch)
        # project the skip to match time‐length & channels
        self.skip = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1, stride=stride, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)       # [batch,128,L_in] → [batch,128,L_out]
        out = self.conv(x)            # → [batch,128,L_out]
        out = self.bn(out)            # normalize
        identity = identity[..., :out.size(-1)]
        out = out + identity          # merge
        out = self.act(out)           # nonlinearity
        return out
