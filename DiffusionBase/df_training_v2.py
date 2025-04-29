import random

import numpy as np
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief
from torchmetrics import R2Score
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from plots import TrainingPlotter
from utils.utils import custom_loss, get_scheduled_k


def forward_diffuse(xt_true, kt, alpha_bar):
    noise = torch.randn_like(xt_true)
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_true.shape[0], xt_true.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    return sqrt_alpha_bar * xt_true + sqrt_one_minus_alpha_bar * noise

def df_training(model, data, validation_data, alpha_bar, K, total_epochs, scaler, loss_type, device):
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3)
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

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3, threshold=0.001)
    #scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = CustomCosineLRScheduler(optimizer, total_epochs=epochs, base_lr=1e-3, min_lr=1e-5)

    plotter = TrainingPlotter()

    for epoch in range(total_epochs):
        epoch_loss, epoch_r2, epoch_r2xt, epoch_smape = 0, 0, 0, 0
        model.train()
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features),
                              device=device)

        for trajectory in data:
            x0 = trajectory.unsqueeze(0).to(device)
            xt = model.fc_project_2_to_hidden(x0)
            optimizer.zero_grad()

            k_min, k_max = get_scheduled_k(epoch, total_epochs, K, min_k=100, max_k=K - 1)
            kt = torch.full((xt.shape[0], xt.shape[1]), random.randint(k_min, k_max), device=device)

            xt_noisy = forward_diffuse(xt, kt, alpha_bar)

            epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
            # zt_prev = zt_updated.detach()

            xt_true = x0
            xt_pred = model.fc_project_xt_output(xt_pred)
            total_loss = custom_loss(epsilon_pred, epsilon_true, xt_pred, xt_true, kt, alpha_bar, epoch, total_epochs, loss_type)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

###################VALIDATION###################
        model.eval()
        val_loss = 0
        r2_eps = R2Score().to(device)
        smape_eps = SymmetricMeanAbsolutePercentageError().to(device)
        r2_xt = R2Score().to(device)
        smape_xt = SymmetricMeanAbsolutePercentageError().to(device)
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features),
                              device=device)

        with torch.no_grad():
            for trajectory in validation_data:
                x0 = trajectory.unsqueeze(0).to(device)
                xt = model.fc_project_2_to_hidden(x0)
                k_min, k_max = get_scheduled_k(epoch, total_epochs, K, min_k=100, max_k=K - 1)
                kt = torch.full((xt.shape[0], xt.shape[1]), random.randint(k_min, k_max), device=device)

                xt_noisy = forward_diffuse(xt, kt, alpha_bar)

                epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

                xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
                zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
                #zt_prev = zt_updated.detach()
###################MODEL EVAL###################

                xt_out = model.fc_project_xt_output(xt_pred)
                val_loss += custom_loss(epsilon_pred, epsilon_true, xt_out, x0, kt, alpha_bar, epoch, total_epochs, loss_type)

                r2_eps_val = r2_eps(epsilon_pred.reshape(-1), epsilon_true.reshape(-1))
                r2_xt_val = r2_xt(xt_out.reshape(-1), x0.reshape(-1))
                smape_eps_val = smape_eps(epsilon_pred.reshape(-1), epsilon_true.reshape(-1))
                smape_xt_val = smape_xt(xt_out.reshape(-1), x0.reshape(-1))

                epoch_r2 += r2_eps_val.item()
                epoch_r2xt += r2_xt_val.item()
                epoch_smape += smape_eps_val.item() * 0.5 + smape_xt_val * 0.5
                ###

        scheduler.step(val_loss / len(validation_data))

        plotter.update_metrics(val_loss / len(validation_data),
                               epoch_r2 / len(validation_data),
                               epoch_r2xt / len(validation_data),
                               epoch_smape / len(validation_data))

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss/len(data):.4f}, Validation Loss: {val_loss / len(validation_data):.4f}")
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch + 1}, LR: {param_group['lr']}")

    plotter.plot_metrics()

def predict(model, test_data, alpha, alpha_bar, K, scaler, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features), device=device)

        for trajectory in test_data:
            x0 = trajectory.unsqueeze(0).to(device)
            xt = model.fc_project_2_to_hidden(x0)
            kt = torch.full((xt.shape[0], xt.shape[1]), random.randrange(0, K)).to(device)

            xt_noisy = forward_diffuse(xt, kt, alpha_bar)

            epsilon_true = (xt_noisy - torch.sqrt(alpha_bar[kt].view(1, -1, 1)) * xt) / torch.sqrt(1 - alpha_bar[kt].view(1, -1, 1))

            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()

            xt_pred_full = model.fc_project_xt_output(xt_pred)

            xt_pred_full = xt_pred_full.squeeze(0).cpu().numpy()
            xt_true_full = x0.squeeze(0).cpu().numpy()
            predictions.append((xt_true_full,
                                xt_pred_full,
                                epsilon_pred.squeeze().cpu().numpy(),
                                epsilon_true.squeeze().cpu().numpy()))

    return predictions
def predict_with_uncertainty(model, test_data, alpha, alpha_bar, K, scaler, device, T=30):
    import numpy as np
    from collections import defaultdict
    import torch

    model.eval()
    enable_dropout(model)

    predictions_by_timestep = defaultdict(list)
    true_by_timestep = {}

    global_timestep = 0

    with torch.no_grad():
        zt_prev = torch.zeros((1,
                               model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features),
                              device=device)

        for i, trajectory in enumerate(test_data):
            x0 = trajectory.unsqueeze(0).to(device)
            xt = model.fc_project_2_to_hidden(x0)
            kt = torch.full((xt.shape[0], xt.shape[1]), random.randrange(0, K), device=device)
            xt_noisy = forward_diffuse(xt, kt, alpha_bar)

            for _ in range(T):
                xt_pred, _, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
                xt_out = model.fc_project_xt_output(xt_pred)
                xt_out_np = xt_out.squeeze(0).cpu().numpy()[:, 0]

                if i < len(test_data) - 1:
                    predictions_by_timestep[global_timestep].append(xt_out_np[0])
                else:
                    for j, val in enumerate(xt_out_np):
                        predictions_by_timestep[global_timestep + j].append(val)

            x0_np = x0.squeeze(0).cpu().numpy()[:, 0]
            if i < len(test_data) - 1:
                true_by_timestep[global_timestep] = x0_np[0]
                global_timestep += 1
            else:
                for val in x0_np:
                    true_by_timestep[global_timestep] = val
                    global_timestep += 1

            zt_prev = zt_prev * 0.7 + zt_updated.detach() * 0.3
            #zt_prev = zt_updated.detach()


    timesteps = sorted(predictions_by_timestep.keys())
    xt_pred_mean = [np.mean(predictions_by_timestep[t]) for t in timesteps]
    xt_pred_std = [np.std(predictions_by_timestep[t]) for t in timesteps]
    xt_true = [true_by_timestep[t] for t in timesteps]

    return {
        "timesteps": timesteps,
        "xt_true": xt_true,
        "xt_pred_mean": xt_pred_mean,
        "xt_pred_std": xt_pred_std
    }

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
