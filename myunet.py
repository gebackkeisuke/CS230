
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyUnet(nn.Module):
    """
    A simple U-Net architecture with one encoder layer, one residual block, and one decoder layer.
    """

    def __init__(self, N_input, N_output, config):
        super().__init__()

        # ---- Encoder ----
        self.enc = nn.Conv2d(
            in_channels=N_input,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )  

        # ---- Bottleneck: residual block ----
        self.res = ResidualBlock(16)

        # ---- Decoder: upsample ----
        self.dec = nn.ConvTranspose2d(
            in_channels=16 + 16, # Skip connection分を足す（Encoderの出力16をconcat）
            out_channels=16,          # skip と同じ16に
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )  

        # ---- Output conv ----
        self.out_conv = nn.Conv2d(
            in_channels=16,
            out_channels=N_output,          # skip と同じ16に
            kernel_size=1,
            stride=1,
            padding=0
        )  

        
    def forward(self, x):
        # ----- Encode -----
        x_enc = self.enc(x)             # (B,16,H/2,W/2)
        # print("x_enc.shape:", x_enc.shape)

        # ----- Residual bottleneck -----
        x_mid = self.res(x_enc)         # (B,16,H/2,W/2)
        # print("x_mid.shape:", x_mid.shape)

        # ----- Decode -----
        x_dec = self.dec(torch.cat([x_mid, x_enc], dim=1))          # (B,16,H,W)
        # print("x_dec.shape:", x_dec.shape)

        # ----- Final output -----
        out = self.out_conv(x_dec)     # (B, N_output, H, W)
        # print("out.shape:", out.shape)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)