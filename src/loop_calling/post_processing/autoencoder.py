import torch.nn as nn


class AdaptiveConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: downsample and encode
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(2)

        # Decoder: upsample and reconstruct
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.act = nn.Sigmoid()  # Assuming input 1D vectors are scaled between 0 and 1

    def forward(self, x):
        input_size = x.size(-1)

        x = self.encoder(x)

        z = self.adaptive_avg_pool(x)  # [batch_size, 2, 2]
        # --------
        # Decoder with shape printing
        # print("Input to decoder:", z.shape)
        # out = z
        # for idx, layer in enumerate(self.decoder):
        #    out = layer(out)
        #    print(f"Shape after decoder layer {idx + 1}: {out.shape}")
        # out = nn.AdaptiveAvgPool1d(input_size)(out)
        # print(f"Final shape after AdaptiveAvgPool1d: {out.shape}")
        # exit()
        # --------
        out = self.decoder(z)
        out = nn.AdaptiveAvgPool1d(input_size)(out)

        return out, z.view(z.size(0), -1)