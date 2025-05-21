import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.GroupNorm(1, out_channels),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class TimestepEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class UNet1D(nn.Module):
    def __init__(self, in_channels, base_channels=64, out_channels=512):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_channels, 64)  # → 64
        self.down2 = ConvBlock(64, 128)  # → 128
        self.bottleneck = ConvBlock(128, 256)  # → 256

        self.up_block2 = ConvBlock(256 + 128, 128)  # → 384 → 128
        self.up_block1 = ConvBlock(128 + 64, 64)  # → 192 → 64

        self.final = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Single level down
        x1 = self.down1(x)  # [B, C, T]
        x2 = self.down2(x1)  # no pooling, just another conv block
        x3 = self.bottleneck(x2)  # deeper features

        # Upsample path (no ConvTranspose)
        x = self.up_block2(torch.cat([x3, x2], dim=1))
        x = self.up_block1(torch.cat([x, x1], dim=1))

        return self.final(x)

class DFBackboneUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()

        self.fc_project_2_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_xt_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # self.unet = nn.Sequential(
        #     nn.Conv1d(hidden_dim * 3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        # )
        self.unet = UNet1D(hidden_dim * 3, base_channels=64, out_channels=hidden_dim)

        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, zt_prev, xt_noisy, k, alpha_bar):
        kt = k.float().unsqueeze(-1)
        normalized_kt = kt / 1000.0
        kt_to_feature = normalized_kt.expand_as(xt_noisy)

        x_in = torch.cat([xt_noisy, kt_to_feature], dim=-1)

        x_in = torch.cat([x_in, zt_prev], dim=-1)
        x_in = x_in.transpose(1, 2)  # [B, hidden+1, T]

        epsilon_pred = self.unet(x_in).transpose(1, 2)  # back to [B, T, hidden]

        zt_updated, _ = self.zt_transition(epsilon_pred)

        # Predict xt
        alpha_t = alpha_bar.gather(0, k.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
        sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
        sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
        xt_pred = (xt_noisy - sqrt_one_minus_alpha_bar * epsilon_pred) / sqrt_alpha_bar

        return xt_pred, epsilon_pred, zt_updated
