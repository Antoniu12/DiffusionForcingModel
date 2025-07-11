import random

import numpy as np
import torch
from torch import nn
from torchmetrics import R2Score
from torchmetrics.regression import MeanAbsolutePercentageError
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from utils.utils import get_scheduled_k

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forward_diffuse(xt_true, kt, alpha_bar):
    noise = torch.randn_like(xt_true)
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_true.shape[0], xt_true.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    return sqrt_alpha_bar * xt_true + sqrt_one_minus_alpha_bar * noise

def compute_epsilon_true(xt_noisy, x_true, kt, alpha_bar):
    alpha_t = alpha_bar.gather(0, kt.view(-1)).view(xt_noisy.shape[0], xt_noisy.shape[1], 1)
    sqrt_alpha_bar = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    epsilon_true = (xt_noisy - sqrt_alpha_bar * x_true) / sqrt_one_minus_alpha_bar
    return epsilon_true

def train_next_token_diffusion(model, data, validation_data, alpha, alpha_bar, K, total_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

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

            loss_xt = (loss_fn(xt_pred, x_target) * 6 + loss_fn(xt_pred[:, :, 0], x_target[:, :, 0]) * 4)# +
                       # loss_fn(xt_pred[:, :, 1], x_target[:, :, 1]) * 2)
            loss_eps = loss_fn(epsilon_pred[:, -1:, :], epsilon_true)

            loss = loss_xt + loss_eps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

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

                loss_xt = loss_fn(xt_pred, x_target) * 10
                loss_eps = loss_fn(epsilon_pred[:, -1:, :], epsilon_true)
                val_loss += (loss_xt + loss_eps).item()

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
    enable_dropout(model)

    predictions = []
    zt_prev = torch.zeros((1, start_offset, model.fc_project_seq_to_hidden.out_features), device=device)

    if isinstance(test_tensor, list):
        test_tensor = torch.stack(test_tensor)
    elif test_tensor.dim() == 2:
        test_tensor = test_tensor.unsqueeze(0)

    with torch.no_grad():
        for t in range(start_offset, test_tensor.shape[1]):
            context = test_tensor[:, t - start_offset : t - 1, :]
            target = test_tensor[:, t, :].unsqueeze(1)

            pred_samples = []
            for _ in range(T):
                k = random.randrange(0, K - 1)
                kt = torch.full((1, 1), k, dtype=torch.long, device=device)

                x_target_noisy = forward_diffuse(target, kt, alpha_bar)
                xt_context = model.encoder(context)
                xt_target_noisy = model.encoder(x_target_noisy)
                xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

                xt_pred, _, _ = model(zt_prev, xt_noisy_full, kt, alpha_bar)
                pred_samples.append(xt_pred.squeeze().cpu().numpy())

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
            context = t[:-1, :].unsqueeze(0).to(device)
            target = t[-1:, :].unsqueeze(0).to(device)
            random_noise = torch.randn_like(target)

            kt = torch.full((1, 1), K - 1, dtype=torch.long, device=device)

            x_target_noisy = forward_diffuse(random_noise, kt, alpha_bar)

            xt_context = model.encoder(context)
            xt_target_noisy = model.encoder(x_target_noisy)
            xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

            xt_pred, epsilon_pred, zt_pred = model(zt_prev, xt_noisy_full, kt, alpha_bar)
            zt_prev = zt_pred.detach()

            preds.append((
                target.squeeze().cpu().numpy(),
                xt_pred.squeeze().cpu().numpy(),
                random_noise.squeeze().cpu().numpy(),
                epsilon_pred.squeeze().cpu().numpy()
            ))

    return preds

def predict_with_random_last_noise_2(
    model, test_tensor, alpha, alpha_bar, K, device, scaler,
    feature_index=0, start_offset=24
):
    model.eval()
    preds = []
    targets = []
    mape = MeanAbsolutePercentageError()
    smape = SymmetricMeanAbsolutePercentageError()
    zt_prev = torch.zeros((1, start_offset, model.fc_project_seq_to_hidden.out_features), device=device)

    with torch.no_grad():
        for t in test_tensor:
            context = t[:-1, :].unsqueeze(0).to(device)
            target = t[-1:, :].unsqueeze(0).to(device)
            random_noise = torch.randn_like(target)
            kt = torch.full((1, 1), K - 1, dtype=torch.long, device=device)

            x_target_noisy = forward_diffuse(random_noise, kt, alpha_bar)
            xt_context = model.encoder(context)
            xt_target_noisy = model.encoder(x_target_noisy)
            xt_noisy_full = torch.cat([xt_context, xt_target_noisy], dim=1)

            xt_pred, epsilon_pred, zt_pred = model(zt_prev, xt_noisy_full, kt, alpha_bar)
            zt_prev = zt_pred.detach()

            pred_val = xt_pred.squeeze().cpu().numpy()[feature_index]
            true_val = target.squeeze().cpu().numpy()[feature_index]

            preds.append(pred_val)
            targets.append(true_val)

    preds = np.array(preds)
    targets = np.array(targets)

    preds = np.clip(preds, 0, 1)

    preds_denorm = []
    targets_denorm = []
    for p, t in zip(preds, targets):
        pred_vec = np.zeros(scaler.scale_.shape)
        true_vec = np.zeros(scaler.scale_.shape)
        pred_vec[feature_index] = p
        true_vec[feature_index] = t
        preds_denorm.append(scaler.inverse_transform([pred_vec])[0][feature_index])
        targets_denorm.append(scaler.inverse_transform([true_vec])[0][feature_index])

    preds_denorm = np.array(preds_denorm)
    targets_denorm = np.array(targets_denorm)

    mae = mean_absolute_error(targets_denorm, preds_denorm)
    mse = mean_squared_error(targets_denorm, preds_denorm)
    r2 = r2_score(targets_denorm, preds_denorm)

    smape_val = smape(torch.tensor(preds_denorm), torch.tensor(targets_denorm))
    mape_val = mape(torch.tensor(preds_denorm), torch.tensor(targets_denorm))
    smape_val = smape_val.item() * 100
    mape_val = mape_val.item() * 100

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "SMAPE": smape_val,
        "MAPE": mape_val,
        "preds": preds_denorm,
        "targets": targets_denorm,
    }
    return metrics

def autoregressive_forecast(
        model, context_seq, alpha, alpha_bar, K, device, steps=24
):
    model.eval()
    preds = []
    feat_dim = context_seq.shape[-1]

    context_seq = context_seq.clone().detach().to(device)
    zt_prev = torch.zeros((1, 24, model.fc_project_seq_to_hidden.out_features), device=device)
    kt_zero = torch.zeros((1, 1), dtype=torch.long, device=device)

    x_target = context_seq[-1:].unsqueeze(0)
    xt_context = model.encoder(context_seq.unsqueeze(0))
    xt_target = model.encoder(x_target)
    xt_noisy_full = torch.cat([xt_context, xt_target], dim=1)
    _, _, zt_pred = model(zt_prev, xt_noisy_full, kt_zero, alpha_bar)
    zt_prev = zt_pred.detach()

    current_window = context_seq[1:].unsqueeze(0).to(device)

    with torch.no_grad():
        for step in range(steps):
            random_token = torch.randn(1, 1, feat_dim).to(device)
            input_window = current_window.clone()

            kt = torch.full((1, 1), K - 1, dtype=torch.long, device=device)
            random_token_diffused = forward_diffuse(random_token, kt, alpha_bar)

            xt_true = model.encoder(input_window)
            xt_target = model.encoder(random_token_diffused)

            xt_hidden = torch.cat([xt_true, xt_target], dim=1)
            xt_pred, _, zt_prev = model(
                zt_prev,
                xt_hidden,
                kt,
                alpha_bar
            )
            zt_prev = zt_prev.detach()

            pred_token = xt_pred[:, -1, :]
            preds.append(pred_token.squeeze(0).cpu())

            current_window = torch.cat([current_window[:, 1:], pred_token.view(1, 1, -1)], dim=1)
    return torch.stack(preds, dim=0)


def teacher_forcing_forecast(
        model, context_seq, ground_truth_seq, alpha, alpha_bar, K, device, steps=24
):
    model.eval()
    preds = []

    context_seq = context_seq.clone().detach().to(device)
    ground_truth_seq = ground_truth_seq.clone().detach().to(device)
    seq_len, feat_dim = context_seq.shape

    zt_prev = torch.zeros((1, seq_len, model.fc_project_seq_to_hidden.out_features), device=device)
    kt_zero = torch.zeros((1, 1), dtype=torch.long, device=device)

    x_target = context_seq[-1:].unsqueeze(0)
    xt_context = model.encoder(context_seq.unsqueeze(0))
    xt_target = model.encoder(x_target)
    xt_noisy_full = torch.cat([xt_context, xt_target], dim=1)
    _, _, zt_pred = model(zt_prev, xt_noisy_full, kt_zero, alpha_bar)
    zt_prev = zt_pred.detach()

    current_window = context_seq[1:].unsqueeze(0).to(device)


    with torch.no_grad():
        for step in range(steps):
            random_token = torch.randn(1, 1, feat_dim).to(device)
            input_window = current_window.clone()

            kt = torch.full((1, 1), K - 1, dtype=torch.long, device=device)
            random_token_diffused = forward_diffuse(random_token, kt, alpha_bar)

            xt_true = model.encoder(input_window)
            xt_target = model.encoder(random_token)

            xt_hidden = torch.cat([xt_true, xt_target], dim=1)
            xt_pred, _, zt_prev = model(
                zt_prev,
                xt_hidden,
                kt,
                alpha_bar,
            )
            zt_prev = zt_prev.detach()

            pred_token = xt_pred[:, -1, :].squeeze(0).cpu()
            preds.append(pred_token)

            true_token = ground_truth_seq[step].view(1, 1, -1).to(device)
            current_window[:, -1:] = true_token

            if step < steps - 1:
                current_window = torch.cat([current_window[:, 1:], true_token], dim=1)

    return torch.stack(preds, dim=0)

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()