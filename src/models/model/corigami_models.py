# Code taken from https://github.com/tanjimin/C.Origami/
import torch
import torch.nn as nn
import src.models.model.blocks as blocks


# Original ConvModel from C.Origami
class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=256):
        super(ConvModel, self).__init__()
        print('Initializing ConvModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        self.decoder = blocks.Decoder(mid_hidden * 2)

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to:
        bs, feat, img_len
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim=1)
        return input_map


# Our adaptation
class ConvTransModelSmall(ConvModel):

    def __init__(self, mid_hidden=128, num_genomic_features=0):
        super(ConvTransModelSmall, self).__init__(num_genomic_features)
        print('Initializing ConvTransModelSmall')
        if num_genomic_features == 0:
            in_channel = 5
            self.encoder = blocks.SmallEncoder(in_channel=in_channel, output_size=mid_hidden, num_blocks=11)
        else:
            self.encoder = blocks.EncoderSplitSmall(num_genomic_features, output_size=mid_hidden, filter_size=5, num_blocks=11)
        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((210, 210))

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.encoder(x)
        x = self.move_feature_forward(x)
        x = self.attn(x)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        x = self.adaptive_pool(x)
        return x
