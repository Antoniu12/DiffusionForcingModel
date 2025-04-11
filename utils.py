import math

import numpy as np
import torch

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

def plotting_preprocess_xt(xt_true_seq, xt_pred_seq, scaler):
    xt_true_flat = np.concatenate(xt_true_seq, axis=0)
    xt_pred_flat = np.concatenate(xt_pred_seq, axis=0)

    xt_true_full = np.zeros((xt_true_flat.shape[0], 7))
    xt_pred_full = np.zeros((xt_pred_flat.shape[0], 7))
    xt_true_full[:, 3] = xt_true_flat
    xt_pred_full[:, 3] = xt_pred_flat

    xt_true_rescaled = scaler.inverse_transform(xt_true_full)[:, 3]
    xt_pred_rescaled = scaler.inverse_transform(xt_pred_full)[:, 3]
    return xt_true_rescaled, xt_pred_rescaled

def plotting_preprocess_epsilon(epsilon_true_all, epsilon_pred_all):
    epsilon_true_avg = [eps.mean(axis=1) for eps in epsilon_true_all]
    epsilon_pred_avg = [eps.mean(axis=1) for eps in epsilon_pred_all]

    epsilon_true_flat = np.concatenate(epsilon_true_avg)
    epsilon_pred_flat = np.concatenate(epsilon_pred_avg)
    return epsilon_true_flat, epsilon_pred_flat