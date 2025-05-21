import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief

from models.GRU import GRUWithLayerNorm


def predict_start_from_noise(xt_noisy, kt, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-3))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-3))
    x0 = (xt_noisy - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    # x_pred = x0.clamp(-1, 1)
    return x0

class DFBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()

        self.fc_project_2_to_hidden = nn.Linear(input_dim, hidden_dim)
        # self.RNN = nn.Sequential(
        #     nn.Linear(hidden_dim * 3, hidden_dim * 3),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_dim * 3, hidden_dim)
        # )
        # self.RNN = nn.LSTM(3 * hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.epsilon_head = nn.Linear(hidden_dim, hidden_dim)
        self.RNN = GRUWithLayerNorm(3 * hidden_dim, hidden_dim)
        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_xt_output = nn.Linear(hidden_dim, input_dim)
        self.fc_project_hidden_to_feature = nn.Linear(hidden_dim, input_dim)


    @staticmethod
    def sinusoidal_embedding(kt, dim=32, max_k=999):
        """
        kt: Tensor of shape [B, T]
        dim: Embedding dimension (must be even)
        max_k: Maximum value for kt (usually K)
        """
        assert dim % 2 == 0
        device = kt.device
        kt = kt.unsqueeze(-1).float()
        freqs = torch.exp(-math.log(max_k) * torch.arange(0, dim, 2, device=device) / dim)
        freqs = freqs.view(1, 1, -1)

        angles = kt * freqs
        embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embed
    def forward(self, zt_prev, xt_noisy, kt, alpha_bar):
        # kt = k.float().unsqueeze(-1)
        # normalized_kt = kt / 1000.0
        # kt_to_feature = normalized_kt.expand_as(xt_noisy)
        # k_embed = self.k_embedding(normalized_kt.unsqueeze(-1).float())
        k_embed = self.sinusoidal_embedding(kt, dim=512)

        input_xt = torch.cat([xt_noisy, k_embed], dim=-1)
        input_epsilon_pred = torch.cat([zt_prev, input_xt], dim=-1)
        rnn_out, _ = self.RNN(input_epsilon_pred)
        epsilon_pred = self.epsilon_head(xt_noisy + rnn_out)
        zt_updated, _ = self.zt_transition(epsilon_pred)
        xt_pred = predict_start_from_noise(xt_noisy.detach(), kt, epsilon_pred, alpha_bar)
        return xt_pred, epsilon_pred, zt_updated

def pretrain_layers(model, data, total_epochs, device):
    optimizer = AdaBelief(
        model.parameters(),
        lr=5e-4,
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        rectify=True,
        weight_decouple=True,
        print_change_log=False
    )

    model.train()
    for epoch in range(total_epochs):
        epoch_loss = 0.0

        for trajectory in data:
            x_true = trajectory.unsqueeze(0).to(device)
            x_projected = model.fc_project_2_to_hidden(x_true)
            x_reconstructed = model.fc_project_xt_output(x_projected)
            loss = F.mse_loss(x_reconstructed, x_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Projection Recon Loss: {epoch_loss / len(data):.6f}")

    for param in model.fc_project_2_to_hidden.parameters():
        param.requires_grad = False
    for param in model.fc_project_xt_output.parameters():
        param.requires_grad = False



