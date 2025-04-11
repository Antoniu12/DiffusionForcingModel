import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import R2Score
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from plots import TrainingPlotter

# def forward_diffuse(x_start, k, noise, alpha_bar):
#     alpha_t = alpha_bar.gather(0, k.view(-1)).view(x_start.shape[0], x_start.shape[1], 1)
#     sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
#     sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
#     return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
def forward_diffuse(x_start, k, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, k.view(-1)).view(x_start.shape[0], x_start.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

def predict_start_from_noise(x_t, k, noise, alpha_bar):
    alpha_t = alpha_bar.gather(0, k.view(-1)).view(x_t.shape[0], x_t.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0 = (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    return x0
def df_training(model, data, validation_data, alpha, alpha_bar, K, epochs):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3, threshold=0.001)
    loss_function = nn.MSELoss()
    plotter = TrainingPlotter()

    for epoch in range(epochs):
        epoch_loss, epoch_r2, epoch_smape = 0, 0, 0
        model.train()
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features))

        for trajectory in data:
            x0 = trajectory.unsqueeze(0)
            xt = model.fc_project_2_to_hidden(x0)
            optimizer.zero_grad()

            # kt = torch.randint(0, K, (xt.shape[0], xt.shape[1]))
            kt = torch.full((xt.shape[0], xt.shape[1]), 200)
            noise = torch.randn_like(xt)
            xt_noisy = forward_diffuse(xt, kt, noise, alpha_bar)

            epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            zt_prev = 0.9 * zt_prev + 0.1 * zt_updated.detach()

            xt_true = x0

            trajectory_loss = loss_function(epsilon_pred, epsilon_true)
            xt_loss = loss_function(model.fc_project_xt_output(xt_pred), x0)
            total_loss = trajectory_loss * 0.5 + xt_loss * 0.5

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        # Validation
        model.eval()
        val_loss = 0
        r2_eps = R2Score()
        smape_eps = SymmetricMeanAbsolutePercentageError()
        r2_xt = R2Score()
        smape_xt = SymmetricMeanAbsolutePercentageError()
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features))

        with torch.no_grad():
            for trajectory in validation_data:
                x0 = trajectory.unsqueeze(0)
                xt = model.fc_project_2_to_hidden(x0)
                # kt = torch.randint(0, K, (xt.shape[0], xt.shape[1]))
                kt = torch.full((xt.shape[0], xt.shape[1]), 200)

                noise = torch.randn_like(xt)
                xt_noisy = forward_diffuse(xt, kt, noise, alpha_bar)

                epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

                xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
                zt_prev = 0.9 * zt_prev + 0.1 * zt_updated.detach()

                ### MODEL EVAL
                epsilon_loss = loss_function(epsilon_pred, epsilon_true)
                xt_loss = loss_function(model.fc_project_xt_output(xt_pred), x0)

                xt_out = model.fc_project_xt_output(xt_pred)

                r2_eps_val = r2_eps(epsilon_pred.reshape(-1), epsilon_true.reshape(-1))
                r2_xt_val = r2_xt(xt_out.reshape(-1), x0.reshape(-1))

                smape_eps_val = smape_eps(epsilon_pred, epsilon_true)
                smape_xt_val = smape_xt(model.fc_project_xt_output(xt_pred), x0)

                val_loss += epsilon_loss.item() * 0.5 + xt_loss.item() * 0.5
                epoch_r2 += r2_eps_val.item() * 0.5 + r2_xt_val * 0.5
                epoch_smape += smape_eps_val.item() * 0.5 + smape_xt_val * 0.5
                ###

        scheduler.step(val_loss / len(validation_data))

        plotter.update_metrics(val_loss / len(validation_data),
                               epoch_r2 / len(validation_data),
                               epoch_smape / len(validation_data))

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss/len(data):.4f}, Validation Loss: {val_loss / len(validation_data):.4f}")
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch + 1}, LR: {param_group['lr']}")

    plotter.plot_metrics()

def predict(model, test_data, alpha, alpha_bar, K, scaler):
    model.eval()
    predictions = []

    with torch.no_grad():
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features))

        for trajectory in test_data:
            x0 = trajectory.unsqueeze(0)
            xt = model.fc_project_2_to_hidden(x0)
            # kt = torch.randint(0, K, (xt.shape[0], xt.shape[1]))
            kt = torch.full((xt.shape[0], xt.shape[1]), 200)

            noise = torch.randn_like(xt)
            xt_noisy = forward_diffuse(xt, kt, noise, alpha_bar)

            epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            zt_prev = 0.9 * zt_prev + 0.1 * zt_updated.detach()

            xt_pred_full = model.fc_project_xt_output(xt_pred)

            xt_pred_full = xt_pred_full.squeeze(0).cpu().numpy()
            xt_true_full = x0.squeeze(0).cpu().numpy()
            predictions.append((xt_true_full,
                                xt_pred_full,
                                epsilon_pred.squeeze().cpu().numpy(),
                                epsilon_true.squeeze().cpu().numpy()))

    return predictions

class DFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()
        self.fc_project_2_to_hidden = nn.Linear(7, hidden_dim)
        self.RNN = nn.RNN(hidden_dim * 2, hidden_dim, num_layers=2, batch_first=True)
        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)
        self.fc_project_hidden_to_xt = nn.Linear(hidden_dim, input_dim)
        self.fc_project_hidden_to_epsilon = nn.Linear(hidden_dim, input_dim)
        # self.fc_project_xt_output = nn.Sequential(
        #     nn.Linear(hidden_dim, 7)
        #     # nn.Sigmoid()
        # )
        self.fc_project_xt_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)
        )
        self.fc_project_hidden_to_feature = nn.Linear(hidden_dim, 7)

    def forward(self, z_t_prev, x_t_noisy, k, alpha_bar):
        input_epsilon_pred = torch.cat([z_t_prev, x_t_noisy], dim=-1)
        output, _ = self.RNN(input_epsilon_pred)
        zt_updated, _ = self.zt_transition(output)
        x_t_pred = predict_start_from_noise(x_t_noisy, k, output, alpha_bar)
        return x_t_pred, output, zt_updated