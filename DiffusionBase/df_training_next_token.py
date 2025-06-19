import random

import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from torch import nn
from torchmetrics import R2Score, SymmetricMeanAbsolutePercentageError

from DiffusionBase.df_training_v2 import forward_diffuse, compute_epsilon_true, enable_dropout
from utils.utils import compute_sampling_step, get_scheduled_k


def train_next_token_diffusion(model, data, validation_data, alpha, alpha_bar, K, total_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Initialize metrics
    val_r2_eps = R2Score().to(device)
    val_r2_xt = R2Score().to(device)
    val_smape = SymmetricMeanAbsolutePercentageError().to(device)

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0
        zt_prev = torch.zeros((1, 24, model.fc_project_seq_to_hidden.out_features), device=device)

        for seq in data:
            seq = seq.to(device)
            x_context = seq[:-1].unsqueeze(0)
            x_target = seq[-1:].unsqueeze(0)

            kmin, kmax = get_scheduled_k(epoch, total_epochs, K)
            kt = torch.full((1, 1), random.randrange(kmin, kmax), dtype=torch.long, device=device)

            x_target_noisy = forward_diffuse(x_target, kt, alpha_bar)
            epsilon_true = compute_epsilon_true(x_target_noisy, x_target, kt, alpha_bar)

            xt_context = model.encoder(x_context)
            xt_target_noisy = model.encoder(x_target_noisy)
            xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

            xt_pred, epsilon_pred, zt_prev = model(zt_prev, xt_noisy_full, kt, alpha_bar)
            zt_prev = zt_prev.detach()

            loss_xt = loss_fn(xt_pred, x_target) * 10
            loss_eps = loss_fn(epsilon_pred[:, -1:, :], epsilon_true)
            loss = loss_xt + loss_eps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        val_r2_eps.reset()
        val_r2_xt.reset()
        val_smape.reset()

        with torch.no_grad():
            for seq in validation_data:
                seq = seq.to(device)
                x_context = seq[:-1].unsqueeze(0)
                x_target = seq[-1:].unsqueeze(0)

                kt = torch.full((1, 1), random.randrange(0, K - 1), dtype=torch.long, device=device)
                x_target_noisy = forward_diffuse(x_target, kt, alpha_bar)
                epsilon_true = compute_epsilon_true(x_target_noisy, x_target, kt, alpha_bar)

                xt_context = model.encoder(x_context)
                xt_target_noisy = model.encoder(x_target_noisy)
                xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

                xt_pred, epsilon_pred, _ = model(zt_prev, xt_noisy_full, kt, alpha_bar)

                # Compute validation loss
                loss_xt = loss_fn(xt_pred, x_target) * 10
                loss_eps = loss_fn(epsilon_pred[:, -1:, :], epsilon_true)
                val_loss += (loss_xt + loss_eps).item()

                # Update metrics
                val_r2_eps(epsilon_pred[:, -1:, :].flatten(), epsilon_true.flatten())
                val_r2_xt(xt_pred.flatten(), x_target.flatten())
                val_smape(xt_pred.flatten(), x_target.flatten())

        print(f"Epoch {epoch + 1}/{total_epochs}, "
              f"Train Loss: {total_loss / len(data):.4f}, "
              f"Val Loss: {val_loss / len(validation_data):.4f}, "
              f"Val R2 (ε): {val_r2_eps.compute().item():.4f}, "
              f"Val R2 (x): {val_r2_xt.compute().item():.4f}, "
              f"Val SMAPE: {val_smape.compute().item():.4f}")


def predict_with_uncertainty_next_token(model, test_tensor, alpha, alpha_bar, K, device, start_offset=24, T=30):
    model.eval()
    enable_dropout(model)  # Enable dropout for uncertainty

    predictions = []
    zt_prev = torch.zeros((1, start_offset, model.fc_project_seq_to_hidden.out_features), device=device)

    # Ensure test_tensor is a tensor and properly shaped
    if isinstance(test_tensor, list):
        test_tensor = torch.stack(test_tensor)  # Converts list of tensors to (batch, time, features)
    elif test_tensor.dim() == 2:
        test_tensor = test_tensor.unsqueeze(0)  # Add batch dim if missing: (1, time, features)

    with torch.no_grad():
        for t in range(start_offset, test_tensor.shape[1]):  # Iterate over time dimension
            # Extract context: [batch, t-start_offset : t-1], target: [batch, t]
            context = test_tensor[:, t - start_offset : t - 1, :]  # (batch, 23, features)
            target = test_tensor[:, t, :].unsqueeze(1)  # (batch, 1, features)

            # Monte Carlo sampling for uncertainty
            pred_samples = []
            for _ in range(T):
                k = random.randrange(0, K - 1)
                kt = torch.full((1, 1), k, dtype=torch.long, device=device)

                x_target_noisy = forward_diffuse(target, kt, alpha_bar)
                xt_context = model.encoder(context)
                xt_target_noisy = model.encoder(x_target_noisy)
                xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)  # (batch, 24, hidden_dim)

                xt_pred, _, _ = model(zt_prev, xt_noisy_full, kt, alpha_bar)
                pred_samples.append(xt_pred.squeeze().cpu().numpy())

            # Compute mean and standard deviation
            pred_mean = np.mean(pred_samples, axis=0)
            pred_std = np.std(pred_samples, axis=0)

            predictions.append((
                target.squeeze().cpu().numpy(),
                pred_mean,
                pred_std
            ))

    return predictions


def rolling_next_token_prediction(model, test_tensor, alpha, alpha_bar, K, device, start_offset=24):

    model.eval()
    preds = []
    zt_prev = torch.zeros((1, start_offset, model.fc_project_seq_to_hidden.out_features), device=device)
    if not isinstance(test_tensor, torch.Tensor):
        test_tensor = torch.tensor(test_tensor, dtype=torch.float32, device=device)
    else:
        test_tensor = test_tensor.to(device)
    with torch.no_grad():
        for t in range(start_offset, len(test_tensor)):
            context = test_tensor[t - (start_offset-1):t].unsqueeze(0).to(device)
            target = test_tensor[t].unsqueeze(0).unsqueeze(1).to(device)
            k = random.randrange(0, K-1)
            kt = torch.full((1, 1), k, dtype=torch.long, device=device)

            x_target_noisy = forward_diffuse(target, kt, alpha_bar)

            xt_context = model.encoder(context)
            xt_target_noisy = model.encoder(x_target_noisy)
            xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

            epsilon_true = compute_epsilon_true(x_target_noisy, target, kt, alpha_bar)

            xt_pred, epsilon_pred, zt_prev = model(zt_prev, xt_noisy_full, kt, alpha_bar)


            preds.append((
                target.squeeze(0).squeeze(0).cpu(),
                xt_pred.squeeze(0).squeeze(0).cpu(),
                epsilon_pred[:, -1, :].squeeze(0).cpu(),
                epsilon_true.squeeze(0).squeeze(0).cpu()
            ))

    return preds


def predict_with_random_last_noise(model, test_tensor, alpha, alpha_bar, K, device, start_offset=24):
    model.eval()
    preds = []
    zt_prev = torch.zeros((1, start_offset, model.fc_project_seq_to_hidden.out_features), device=device)

    with torch.no_grad():
        for t in test_tensor:
            # Get context and true target
            # t.unsqueeze(0)
            context = t[:-1, :].unsqueeze(0).to(device)  # (1, seq_len-1, features)
            target = t[-1:, :].unsqueeze(0).to(device)
            # Generate full random noise for target
            random_noise = torch.randn_like(target)  # (batch, 1, features)

            # Use maximum noise level (K-1)
            kt = torch.full((1, 1), K - 1, dtype=torch.long, device=device)

            # Diffuse the random noise
            x_target_noisy = forward_diffuse(random_noise, kt, alpha_bar)

            # Encode context and noisy target
            xt_context = model.encoder(context)
            xt_target_noisy = model.encoder(x_target_noisy)
            xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

            # Predict denoised output
            xt_pred, epsilon_pred, zt_prev = model(zt_prev, xt_noisy_full, kt, alpha_bar)

            preds.append((
                target.squeeze().cpu().numpy(),  # Original true value
                xt_pred.squeeze().cpu().numpy(),  # Predicted denoised value
                random_noise.squeeze().cpu().numpy(),  # Random noise used as input
                epsilon_pred.squeeze().cpu().numpy()  # Predicted noise
            ))

    return preds

def autoregressive_forecast_with_pred_as_next(
    model, context_seq, alpha, alpha_bar, K, device, steps=24
):
    """
    context_seq: (seq_len, features) torch.Tensor (the initial warm-up/context)
    Returns: list of predicted values, one per forecast step
    """
    model.eval()
    preds = []

    # Setup
    context_seq = context_seq.clone().detach().to(device)
    seq_len, feat_dim = context_seq.shape
    zt_prev = torch.zeros((1, seq_len, model.fc_project_seq_to_hidden.out_features), device=device)
    current_window = context_seq.unsqueeze(0).to(device)  # (1, seq_len, features)

    with torch.no_grad():
        for step in range(steps):
            # Use the last window as input, add noise to the last token
            kt = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            kt[0, -1] = K-1  # Only last token gets max noise

            # Encode
            xt_hidden = model.encoder(current_window)  # (1, seq_len, H)
            xt_noisy = xt_hidden.clone()
            xt_noisy[:, -1, :] = forward_diffuse(
                xt_hidden[:, -1, :].unsqueeze(1),  # (1, 1, H)
                torch.full((1, 1), K-1, dtype=torch.long, device=device),  # (1, 1)
                alpha_bar
            ).squeeze(1)  # (1, H)

            xt_pred, epsilon_pred, zt_prev = model(zt_prev, xt_noisy, torch.full((1, 1), K-1, dtype=torch.long, device=device), alpha_bar)
            zt_prev = zt_prev.detach()

            pred_token = xt_pred[:, -1, :].squeeze(0).cpu()
            preds.append(pred_token)

            next_window = torch.cat([
                current_window[:, 1:, :].cpu(),
                pred_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
            current_window = next_window.to(device)

    return torch.stack(preds, dim=0)
