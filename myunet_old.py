
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyUnet(nn.Module):
    def __init__(self, N_input, N_output, config):
        super().__init__()
        
        self.enc_layers = nn.ModuleList()
        in_ch = N_input
        # IN_CH = []
        for layer_cfg in config["Unet"]["encoder"]:
            self.enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = in_ch,
                        out_channels= layer_cfg["output_channels"],
                        kernel_size = layer_cfg["kernel_size"],
                        stride      = layer_cfg["stride"],
                        padding     = layer_cfg["padding"]
                    ),
                    nn.BatchNorm2d(layer_cfg["output_channels"]),
                    nn.ReLU(inplace=True)
                    )
            )
            in_ch = layer_cfg["output_channels"]
            # IN_CH.append(in_ch)

        self.res_blocks = nn.ModuleList()
        for block_cfg in config["Unet"]["residual_blocks"]:
            self.res_blocks.append(ResidualBlock(block_cfg))

        self.dec_layers = nn.ModuleList()
        prev_ch = config["Unet"]["residual_blocks"][-1]["channels"]

        # for layer_cfg, in_ch in zip(config["Unet"]["decoder"], reversed(IN_CH)):
        #     out_ch = layer_cfg["out_channels"]

        #     self.dec_layers.append(
        #         nn.ConvTranspose2d(
        #             in_channels    = prev_ch + in_ch,  # skip connection の分を加える
        #             out_channels   = out_ch,
        #             kernel_size    = layer_cfg["kernel_size"],
        #             stride         = layer_cfg["stride"],
        #             padding        = layer_cfg["padding"],
        #             output_padding = 0, # assuming no output padding
        #         )
        #     )
        #     prev_ch = out_ch

        for layer_cfg in config["Unet"]["decoder"]:
            out_ch = layer_cfg["out_channels"]
            self.dec_layers.append(
                nn.ConvTranspose2d(
                    in_channels = prev_ch*2,   # skip は forward で concat
                    out_channels = out_ch,
                    kernel_size = layer_cfg["kernel_size"],
                    stride = layer_cfg["stride"],
                    padding = layer_cfg["padding"],
                    output_padding = 0
                )
            )
            prev_ch = out_ch

        # Output layer
        layer_cfg = config["Unet"]["output_layer"]
        self.out_conv   = nn.Conv2d(
            in_channels = prev_ch,
            out_channels= layer_cfg["out_channels"],
            kernel_size = layer_cfg["kernel_size"],
            stride      = layer_cfg["stride"],
            padding     = layer_cfg["padding"]
        )

    def forward(self, x):
        print("Input x.shape:", x.shape)
        # --- Encoder ---
        x_enc = x
        skips = []
        for layer in self.enc_layers:
            # x = F.relu(layer(x))
            x_enc = layer(x_enc)
            skips.append(x_enc)
            print("x_enc.shape:", x_enc.shape)

        x_res = x_enc
        # --- Residual Blocks ---
        for res_block in self.res_blocks:
            x_res = res_block(x_res)
            print("x_res.shape:", x_res.shape)

        x_dec = x_res
        # --- Decoder ---
        # skip connection は encoder 出力を逆順で使用
        for dec_layer, skip in zip(self.dec_layers, reversed(skips)):
            x_dec = dec_layer(x_dec)
            print("x_dec before concat.shape:", x_dec.shape)
            x_dec = torch.cat([x_dec, skip], dim=1)  # チャネル方向に結合
            print("x_dec after concat.shape:", x_dec.shape)
        
        # for dec_layer, skip, conv_layer in zip(self.dec_layers, reversed(skips), self.dec_conv_layers):
        #     x_dec = dec_layer(x_dec)
        #     print("x_dec after dec_layer.shape:", x_dec.shape)
        #     x_dec = torch.cat([x_dec, skip], dim=1)  # チャンネル増加
        #     x_dec = conv_layer(x_dec)                # 2*channels → channels
        #     print("x_dec after concat and conv.shape:", x_dec.shape)

        # --- Output layer ---
        out = self.out_conv(x_dec)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_bn = cfg.get("use_bn", True)
        
        self.conv1 = nn.Conv2d(
            in_channels  = cfg["channels"],
            out_channels = cfg["channels"],
            kernel_size  = cfg.get("kernel_size", 3),
            stride       = cfg.get("stride", 1),
            padding      = cfg.get("padding", 1)
        )
        self.bn1 = nn.BatchNorm2d(cfg["channels"]) if self.use_bn else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            in_channels  = cfg["channels"],
            out_channels = cfg["channels"],
            kernel_size  = cfg.get("kernel_size", 3),
            stride       = cfg.get("stride", 1),
            padding      = cfg.get("padding", 1)
        )
        self.bn2 = nn.BatchNorm2d(cfg["channels"]) if self.use_bn else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
