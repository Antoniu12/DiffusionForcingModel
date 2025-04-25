import math

import numpy as np
import pysdtw
import torch
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-8, max=0.999)
    return betas

def get_alphas(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def plotting_preprocess_epsilon(epsilon_true_all, epsilon_pred_all):
    epsilon_true_avg = [eps.mean(axis=1) for eps in epsilon_true_all]
    epsilon_pred_avg = [eps.mean(axis=1) for eps in epsilon_pred_all]

    epsilon_true_flat = np.concatenate(epsilon_true_avg)
    epsilon_pred_flat = np.concatenate(epsilon_pred_avg)
    return epsilon_true_flat, epsilon_pred_flat

def custom_loss(epsilon_pred, epsilon_true, xt_pred, xt_true, kt, alpha_bar, epoch, epochs, loss):
    if loss == "snr":
        snr = alpha_bar[kt] / (1.0 - alpha_bar[kt])
        snr = torch.clamp(snr, min=0.01, max=10.0).unsqueeze(-1)

        x0_loss = snr * (xt_pred - xt_true) ** 2
        x0_loss[..., 3] *= 5
        epsilon_loss = snr * (epsilon_pred - epsilon_true) ** 2

        trajectory_loss = epsilon_loss.mean()
        xt_loss = x0_loss.mean()
        return 0.3 * trajectory_loss + 0.7 * xt_loss

    elif loss == "dinamic":
        xt_weight, epsilon_weight = get_dynamic_loss_weights(epoch, epochs)
        xt_loss = ((xt_pred[:, :, 3] - xt_true[:, :, 3]) ** 2).mean() * 0.7 + ((xt_pred - xt_true) ** 2).mean() * 0.3
        epsilon_loss = ((epsilon_pred - epsilon_true) ** 2).mean()
        return xt_weight * xt_loss + epsilon_weight * epsilon_loss
    else:
        x0_loss = F.mse_loss(xt_pred, xt_true)
        epsilon_loss = F.mse_loss(epsilon_pred, epsilon_true)
        return 0.3 * epsilon_loss + 0.7 * x0_loss

def get_scheduled_k(epoch, total_epochs, K, min_k=50, max_k=None):
    max_k = max_k if max_k is not None else K - 1
    progress = epoch / total_epochs
    k_start = int(min_k + progress * (max_k - min_k))
    return k_start, max_k

def get_dynamic_loss_weights(epoch, total_epochs, sharpness=10):
    p = epoch / total_epochs
    p_tensor = torch.tensor(p)
    transition = torch.sigmoid(sharpness * (p_tensor - 0.5))
    epsilon_weight = 0.3 * (1 - transition) + 0.1 * transition
    x0_weight = 1.0 - epsilon_weight

    return x0_weight.item(), epsilon_weight.item()
