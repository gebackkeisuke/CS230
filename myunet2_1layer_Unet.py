
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

        # --- Residual Block を 1 層 ---
        self.res = ResidualBlock(16)
        
        # --- Decoder: Up sampling layer ---
        self.up = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=N_output,
            kernel_size=4,
            stride=2,
            padding=1
        )


    def forward(self, x):
        x = self.conv1(x)

        x = self.res(x)
        
        out = self.up(x)
        
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