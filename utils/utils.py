import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from properscoring import crps_gaussian


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a cosine-beta schedule for the diffusion process.

    :param timesteps: The maximum level of noise / Total number of diffusion steps.
    :param s: Offset to prevent betas of becoming 0.
    :return: A tensor of shape (timesteps,) containing beta values for each diffusion step.

     Description:
        - Uses a cosine curve to smoothly control the cumulative product of alphas.
        - The alphas_cumprod curve follows a scaled squared cosine shape.
        - Beta at each step is defined as 1 - (next cumulative alpha / current cumulative alpha).
        - Betas are clamped between [1e-8, 0.999] for numerical stability.
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, min=1e-8, max=0.999)
    return betas
def get_alphas(betas):
    """
    Computes alphas and cumulative product of alphas from beta schedule.

    :param betas: Tensor of beta values from the beta scheduler.
    :return:
        - alpha: Tensor, alphas at each timestep, where alpha = 1 - beta
        - alpha_bars: Tensor, cumulative product of alphas

    Description:
        - Alphas represent the amount of "signal" preserved after each diffusion step.
        - Alpha_bars represent how much total "signal" is preserved up to time t.
    """

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars

def get_scheduled_k(epoch, total_epochs, K, min_k=0, max_k=None):
    max_k = max_k if max_k is not None else K - 1

    progress = min(epoch / total_epochs, 1.0)

    k_mid = (min_k + max_k) / 2
    k_range = (max_k - min_k) / 2

    k_center = int(k_mid + k_range * np.cos(np.pi * (1 - progress)))

    margin = int((1 - progress) * k_range * 0.5)
    k_min_sched = max(min_k, k_center - margin)
    k_max_sched = min(max_k, k_center + margin)

    return k_min_sched, k_max_sched
    # return K-1, K-1

def compute_crps(ground_truth, mean_prediction, std_prediction):
    """
    ground_truth: numpy array (shape [N,])
    mean_prediction: numpy array (shape [N,])
    std_prediction: numpy array (shape [N,])
    """
    crps = crps_gaussian(ground_truth, mean_prediction, std_prediction)
    return crps.mean()

def save_model(model, save_dir="saved_models", prefix="df_model"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved successfully at: {filepath}")

def plotting_preprocess_epsilon(epsilon_true_all, epsilon_pred_all):
    epsilon_true_avg = [eps.mean() for eps in epsilon_true_all]
    epsilon_pred_avg = [eps.mean() for eps in epsilon_pred_all]

    epsilon_true_flat = np.array(epsilon_true_avg)
    epsilon_pred_flat = np.array(epsilon_pred_avg)
    return epsilon_true_flat, epsilon_pred_flat

def flatten_overlapping_windows_batched(window_list):
    output = [window_list[0].squeeze(0)]
    for window in window_list[1:]:
        output.append(window[-1, :].squeeze(0))
    return torch.cat(output, dim=0)

def flatten_overlapping_windows(window_list):
    output = [window_list[0]]
    for window in window_list[1:]:
        output.append(window[-1:, :])
    return torch.cat(output, dim=0)

def flatten_overlapping_windows_preds(window_list):
    output = []
    for pred_tuple in window_list:
        arr = np.array(pred_tuple[1])
        if arr.ndim == 2:
            output.append(arr[-1])
        else:
            output.append(arr)
    return np.stack(output, axis=0)

def flatten_overlapping_windows_targets(window_list):
    output = []
    for arr in window_list:
        arr = np.array(arr)
        if arr.ndim == 2:
            output.append(arr[-1])
        else:
            output.append(arr)
    return np.stack(output, axis=0)

