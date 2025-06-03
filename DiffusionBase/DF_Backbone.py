# import math
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from adabelief_pytorch import AdaBelief
#
# from models.GRU import GRUWithLayerNorm
#
#
# def predict_start_from_noise(xt_noisy, kt, noise, alpha_bar):
#     alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
#     sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-9))
#     sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-9))
#     x0 = (xt_noisy - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
#
#     return x0
#
# class DFBackbone(nn.Module):
#     def __init__(self, input_dim, hidden_dim, seq_dim):
#         super().__init__()
#
#         self.encoder = nn.Linear(input_dim, hidden_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, input_dim)
#         )
#         self.epsilon_head = nn.Linear(hidden_dim, hidden_dim)
#         self.xt_prediction = nn.Sequential(
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.xt_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#         self.RNN = GRUWithLayerNorm(2 * hidden_dim, hidden_dim)
#         self.zt_transition = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.zt_gate_layer = nn.Linear(hidden_dim, hidden_dim)
#
#         self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
#
#
#     @staticmethod
#     def sinusoidal_embedding(kt, dim=32, max_k=999):
#         assert dim % 2 == 0
#         device = kt.device
#         kt = kt.unsqueeze(-1).float()
#         freqs = torch.exp(-math.log(max_k) * torch.arange(0, dim, 2, device=device) / dim)
#         freqs = freqs.view(1, 1, -1)
#
#         angles = kt * freqs
#         embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
#         return embed
#
#     def forward(self, zt_prev, xt_noisy, kt, alpha_bar):
#
#         k_embed = self.sinusoidal_embedding(kt, dim=512)
#
#         input_epsilon_pred = torch.cat([xt_noisy, k_embed], dim=-1)
#         rnn_out, _ = self.RNN(input_epsilon_pred)
#         epsilon_pred = self.epsilon_head(xt_noisy + rnn_out)
#
#         xt_pred = predict_start_from_noise(xt_noisy, kt, epsilon_pred, alpha_bar)
#         xt_pred_output = self.decoder(xt_pred).clamp(0, 1)
#
#         zt_updated, _ = self.zt_transition(xt_pred)
#
#         return xt_pred_output, epsilon_pred, zt_updated
#
# def pretrain_layers(model, data, total_epochs, device, sqrt_alpha_bar_K=0.03):
#     from torch.utils.data import DataLoader
#     import torch.nn.functional as F
#
#     decoder = model.decoder
#     encoder = model.encoder
#     decoder.train()
#
#     optimizer = AdaBelief(
#         decoder.parameters(),
#         lr=5e-4,
#         eps=1e-16,
#         betas=(0.9, 0.999),
#         weight_decay=1e-2,
#         rectify=True,
#         weight_decouple=True,
#         print_change_log=False
#     )
#
#     for epoch in range(total_epochs):
#         epoch_loss = 0.0
#         count = 0
#
#         for trajectory in data:
#             x0 = trajectory.to(device).unsqueeze(0)
#             x_encoded = encoder(x0)
#             input_scaled = x_encoded * sqrt_alpha_bar_K
#             target = x0
#
#             output = decoder(input_scaled)
#             loss = F.mse_loss(output, target)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             count += 1
#
#         print(f"Epoch {epoch + 1}, Decoder Pretrain Loss: {epoch_loss / count:.6f}")
#
#     for param in decoder.parameters():
#         param.requires_grad = True
#
#
#
#
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

class DFBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.RNN = nn.RNN(2 * hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_xt_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        self.xt_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.epsilon_head = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def sinusoidal_embedding(kt, dim=32, max_k=999):
        assert dim % 2 == 0
        device = kt.device
        kt = kt.unsqueeze(-1).float()
        freqs = torch.exp(-math.log(max_k) * torch.arange(0, dim, 2, device=device) / dim)
        freqs = freqs.view(1, 1, -1)

        angles = kt * freqs
        embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embed
    def forward(self, zt_prev, xt_noisy, k, alpha_bar):
        kt = k.float().unsqueeze(-1)
        normalized_kt = kt / 1000.0
        kt_to_feature = normalized_kt.expand_as(xt_noisy)
        k_embed = self.sinusoidal_embedding(k, dim=512)
        input_xt = torch.cat([xt_noisy, kt_to_feature], dim=-1)

        rnn_out, _ = self.RNN(input_xt)
        epsilon_pred = self.epsilon_head(xt_noisy + rnn_out)

        xt_pred = predict_start_from_noise(xt_noisy, k, epsilon_pred, alpha_bar)
        xt_input = torch.cat([xt_pred, zt_prev], dim=-1)
        xt_pred = self.xt_head(xt_input)
        # xt_pred.clamp_(-1.5, 1.5)
        zt_updated, _ = self.zt_transition(xt_pred)

        xt_pred = self.fc_project_xt_output(xt_pred)
        xt_pred = torch.sigmoid(xt_pred)
        return xt_pred, epsilon_pred, zt_updated

def pretrain_layers(model, data, total_epochs, device, sqrt_alpha_bar_K=0.03):
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    encoder = model.encoder
    decoder = model.fc_project_xt_output
    model.train()

    optimizer = AdaBelief(
        decoder.parameters(),
        lr=5e-4,
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        rectify=True,
        weight_decouple=True,
        print_change_log=False
    )

    for epoch in range(total_epochs):
        epoch_loss = 0.0
        count = 0

        for trajectory in data:
            x0 = trajectory.to(device).unsqueeze(0)
            x_encoded = encoder(x0)
            input_scaled = x_encoded * sqrt_alpha_bar_K
            target = x0

            output = decoder(input_scaled)
            loss = F.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        print(f"Epoch {epoch + 1}, Decoder Pretrain Loss: {epoch_loss / count:.6f}")
        for param in encoder.parameters():
            param.requires_grad = True
        for param in encoder.parameters():
            param.requires_grad = True
