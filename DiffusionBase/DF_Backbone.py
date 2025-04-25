import torch
import torch.nn as nn

class DFBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()
        self.fc_project_2_to_hidden = nn.Linear(7, hidden_dim)
        self.RNN = nn.RNN(hidden_dim * 2, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_xt_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 7)
        )
        self.fc_project_hidden_to_feature = nn.Linear(hidden_dim, 7)

    def forward(self, z_t_prev, x_t_noisy, k, alpha_bar):
        input_epsilon_pred = torch.cat([z_t_prev, x_t_noisy], dim=-1)
        output, _ = self.RNN(input_epsilon_pred)
        zt_updated, _ = self.zt_transition(output)
        x_t_pred = predict_start_from_noise(x_t_noisy, k, output, alpha_bar)
        x_t_pred.clamp_(0, 1)
        return x_t_pred, output, zt_updated

def predict_start_from_noise(x_t, k, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, k.view(-1)).view(x_t.shape[0], x_t.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0 = (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    return x0
