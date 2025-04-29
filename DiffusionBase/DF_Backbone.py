import torch
import torch.nn as nn
import torch.nn.functional as F

def predict_start_from_noise(xt_noisy, kt, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0 = (xt_noisy - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    return x0

class DFBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()

        self.fc_project_2_to_hidden = nn.Linear(input_dim, hidden_dim)

        self.RNN = nn.RNN(3 * hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, dropout=0.2)

        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_xt_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        self.fc_project_hidden_to_feature = nn.Linear(hidden_dim, input_dim)

    def forward(self, z_t_prev, xt_noisy, k, alpha_bar):
        kt = k.float().unsqueeze(-1)
        normalized_kt = kt / 1000.0
        kt_to_feature = normalized_kt.repeat(1, 1, xt_noisy.size(-1))

        input_xt = torch.cat([xt_noisy, kt_to_feature], dim=-1)
        input_epsilon_pred = torch.cat([z_t_prev, input_xt], dim=-1)

        epsilon_pred, _ = self.RNN(input_epsilon_pred)
        zt_updated, _ = self.zt_transition(epsilon_pred)

        xt_pred = predict_start_from_noise(xt_noisy, k, epsilon_pred, alpha_bar)
        xt_pred.clamp_(0, 1)
        return xt_pred, epsilon_pred, zt_updated
