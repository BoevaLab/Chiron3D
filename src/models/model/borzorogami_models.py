import torch
import torch.nn as nn
import src.models.model.blocks as blocks
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from einops import rearrange

def diagonalize_small(x):
    x_i = x.unsqueeze(2).repeat(1, 1, 105, 1)
    x_j = x.unsqueeze(3).repeat(1, 1, 1, 105)
    input_map = torch.cat([x_i, x_j], dim=1)
    return input_map


def move_feature_forward(x):
    # Input: (B, L, C) -> Output: (B, C, L)
    return x.transpose(1, 2).contiguous()


def get_borzoi_model(local: bool, model_type: str):
    assert model_type in ["borzoi", "flashzoi"], "Invalid model type. Choose 'borzoi' or 'flashzoi'."
    cfg = BorzoiConfig.from_pretrained(f"/cluster/work/boeva/shoenig/{model_type}")
    cfg.return_center_bins_only = False                 # forces 16,352 bins
    borzoi = Borzoi.from_pretrained(f"/cluster/work/boeva/shoenig/{model_type}", config=cfg)
    return borzoi
    #if local:
    #    return Borzoi.from_pretrained(f'johahi/{model_type}-replicate-0')
    #return Borzoi.from_pretrained(f"/cluster/work/boeva/shoenig/{model_type}")



# Main implementation - used for initial FULL LORA and LORA TF runs - the the LORA everything except Conv Tower
class BorzoiOrogami(nn.Module):

    def __init__(self, mid_hidden=128, local=False, model_type="borzoi"):
        super().__init__()

        self.borzoi = get_borzoi_model(local, model_type)

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
        #x = torch.relu(x)
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
        identity = self.skip(x)            # [batch,128,L_in] → [batch,128,L_out]
        out = self.conv(x)            # → [batch,128,L_out]
        out = self.bn(out)             # normalize
        identity = identity[..., :out.size(-1)]
        out = out + identity          # merge
        out = self.act(out)           # nonlinearity
        return out

class BorzoiOrogamiCTCF(nn.Module):

    def __init__(self, mid_hidden=128, local=False, model_type="borzoi"):
        super().__init__()

        self.borzoi = get_borzoi_model(local, model_type)

        for param in self.borzoi.parameters():
            param.requires_grad = False
        self.borzoi.eval()

        self.activation = nn.ReLU()
        self.projector = nn.Conv1d(1536, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.projector_CTCF = nn.Conv1d(128, 128, kernel_size=24, stride=1, padding=0, bias=True)
        #self.length_reducer = nn.Conv1d(6144, 105, kernel_size=1, stride=1, padding=0, bias=True)
        self.length_reducer = nn.Sequential(
            ResidualDownBlock(128, kernel_size=8, stride=4),  # 6144 → 1535
            ResidualDownBlock(128, kernel_size=8, stride=4),  # 1535 →  382
            ResidualDownBlock(128, kernel_size=8, stride=3),  # 382 →  125
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=21, stride=1, padding=0)  # 125 →  105
        )

        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden*2, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 4, hidden=128)

        # CTCF ENCODING

        self.conv_start_ctcf = nn.Sequential(
            nn.Conv1d(1, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        hiddens = [16, 16, 16, 16, 32, 32, 64, 64, 64, 64, 128]
        hidden_ins = [16, 16, 16, 16, 16, 32, 32, 64, 64, 64, 64]

        self.res_blocks_epi = self.get_res_blocks(11, hidden_ins, hiddens)

    def forward(self, x):
        dna = x[:, :4, :]
        feat = x[:, 4:, :]
        feat = self.conv_start_ctcf(feat)  # b 8 len/2
        feat = self.res_blocks_epi(feat)  # b c l
        feat = self.projector_CTCF(feat)  # b c l
        x = self.borzoi.get_embs_after_crop(dna)
        x = self.projector(x)
        x = self.activation(x)
        x = self.length_reducer(x)
        comb = torch.cat((feat, x), dim=1)
        comb = rearrange(comb, 'b c l -> b l c')
        comb = self.attn(comb)
        comb = rearrange(comb, 'b l c -> b c l')
        comb = diagonalize_small(comb)
        comb = self.decoder(comb).squeeze(1)
        comb = torch.relu(comb)
        return comb

    def get_res_blocks(self, n, his, hs):
        block_list = []
        for i, h, hi in zip(range(n), hs, his):
            block_list.append(blocks.ConvBlock(5, hidden_in=hi, hidden=h))
        res_blocks = nn.Sequential(*block_list)
        return res_blocks
