
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyUnet(nn.Module):
    def __init__(self, N_input, N_output, config):
        super().__init__()

        # --- Encoder の1層だけ ---
        self.conv1 = nn.Conv2d(
            in_channels=N_input,   # 入力チャネル数
            out_channels=16,       # 16フィルター
            kernel_size=3,
            stride=2,
            padding=1              # 出力サイズを半分に
        )
        
        # --- Decoder: Up sampling layer ---
        self.up = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=N_output,
            kernel_size=4,
            stride=2,
            padding=1
        )


    def forward(self, x):
        # print("Input:", x.shape)

        x = self.conv1(x)
        # print("After conv1:", x.shape)   # → batch, 16, 100, 100

        out = self.up(x)
        # print("After up:", out.shape)    # → batch, 1, 200, 200

        return out    