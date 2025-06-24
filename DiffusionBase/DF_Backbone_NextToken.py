import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief


def predict_start_from_noise(xt_noisy, kt, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0 = (xt_noisy - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    return x0

class DFBackbone_NextToken(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.epsilon_pipeline = nn.RNN(2 * hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

        self.zt_transition = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        self.xt_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.epsilon_head = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def sinusoidal_embedding(kt, dim=42, max_k=999):
        assert dim % 2 == 0
        device = kt.device
        kt = kt.unsqueeze(-1).float()
        freqs = torch.exp(-math.log(max_k) * torch.arange(0, dim, 2, device=device) / dim)
        freqs = freqs.view(1, 1, -1)

        angles = kt * freqs
        embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embed

    def forward(self, zt_prev, xt_noisy, k, alpha_bar):
        B, T, H = xt_noisy.shape
        kt = k.float().unsqueeze(-1)
        normalized_kt = kt / 1000.0
        kt_to_feature = normalized_kt.expand(B, 1, H)
        full_kt_features = torch.zeros_like(xt_noisy)
        full_kt_features[:, -1:, :] = kt_to_feature

        input_xt = torch.cat([xt_noisy, full_kt_features], dim=-1)
        rnn_out, _ = self.epsilon_pipeline(input_xt)

        epsilon_pred = self.epsilon_head(xt_noisy[:, -1:, :] + rnn_out[:, -1:, :])

        xt_input = torch.cat([xt_noisy[:, -1:, :], zt_prev[:, -1:, :], epsilon_pred], dim=-1)
        xt_hidden = self.xt_head(xt_input)
        zt_updated, _ = self.zt_transition(xt_hidden)

        xt_pred = self.decoder(xt_hidden)

        return xt_pred, epsilon_pred, zt_updated

