import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from properscoring import crps_gaussian

from DiffusionBase.DF_Backbone import predict_start_from_noise


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

# def compute_sampling_step(xk, epsilon, kt, alpha, alpha_bar, add_noise=True, eta=1.0):
#     alpha_bar_t = alpha_bar.gather(0, kt.view(-1)).view(xk.shape[0], xk.shape[1], 1)
#     alpha_t = alpha.gather(0, kt.view(-1)).view(xk.shape[0], xk.shape[1], 1)
#     sqrt_alpha = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
#     sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
#     beta_t = 1.0 - alpha_t
#     noise = torch.randn_like(xk) if add_noise else 0.0
#     x_prev = (1 / sqrt_alpha) * (xk - ((1 - alpha_t) / sqrt_one_minus_alpha_bar) * epsilon)
#     x_prev += eta * torch.sqrt(beta_t) * noise
#
#     return x_prev
def compute_sampling_step(xk, epsilon, kt, alpha_bar, add_noise=True, eta=1.0):
    alpha_bar_t = alpha_bar.gather(0, kt.view(-1)).view(xk.shape[0], xk.shape[1], 1)
    alpha_bar_prev = alpha_bar[torch.clamp(kt - 1, min=0)].view(xk.shape[0], xk.shape[1], 1)
    alpha_t = alpha_bar_t / (alpha_bar_prev + 1e-8)
    beta_t = 1.0 - alpha_t

    if add_noise:
        noise = torch.randn_like(xk)
        sigma_t = eta * torch.sqrt(beta_t)
    else:
        noise = 0.0
        sigma_t = 0.0
    x_prev = (1.0 / torch.sqrt(alpha_t)) * (xk - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)) * epsilon) \
             + sigma_t * noise
    # x_prev = x_prev.clamp(0, 1)

    return x_prev
def custom_loss(epsilon_pred, epsilon_true, xt_pred, xt_true, xt_pred_hidden, xt_true_hidden, kt, alpha_bar, epoch, total_epochs, loss_type):
    if loss_type == "snr":
        adaptive_weight = get_dynamic_loss_weights(epoch, total_epochs)
        snr = alpha_bar[kt] / (1.0 - alpha_bar[kt])
        snr = torch.clamp(snr, min=0.01, max=10.0).unsqueeze(-1)
        xt_loss = snr * (xt_pred - xt_true) ** 2
        xt_loss[..., 3] *= 5
        epsilon_loss = snr * (epsilon_pred - epsilon_true) ** 2
        epsilon_loss = epsilon_loss.mean()
        xt_loss = xt_loss.mean()
        return (1-adaptive_weight) * epsilon_loss + adaptive_weight * xt_loss
    elif loss_type == "mse+l1":
        adaptive_weight = get_dynamic_loss_weights(epoch, total_epochs)
        xt_loss_std = torch.std(xt_true) + 1e-8
        epsilon_loss_std = torch.std(epsilon_true) + 1e-8
        normalized_xt_loss = (F.mse_loss(xt_pred[:, :, 0], xt_true[:, :, 0]) * 0.7 + F.mse_loss(xt_pred, xt_true) * 0.3) / xt_loss_std
        normalized_epsilon_loss = F.smooth_l1_loss(epsilon_pred, epsilon_true) / epsilon_loss_std
        total_loss = adaptive_weight * normalized_xt_loss + (1 - adaptive_weight) * normalized_epsilon_loss
        return total_loss
    elif loss_type =="huber":
        huber_loss = F.huber_loss(xt_pred, xt_true, delta=1.0)
        mse_loss = F.mse_loss(xt_pred, xt_true)
        epsilon_loss = F.l1_loss(epsilon_pred, epsilon_true)
        return 0.6 * huber_loss + 0.3 * mse_loss + 0.1 * epsilon_loss
    elif loss_type == "spike_and_small_masked":
        adaptive_weight = get_dynamic_loss_weights(epoch, total_epochs)
##############################################################################XT
        xt_loss_std = torch.std(xt_true[:, :, :2]) + 1e-8
        xt_loss = (F.mse_loss(xt_pred[:, :, 0], xt_true[:, :, 0]) * 0.5 +
                   F.mse_loss(xt_pred[:, :, 1], xt_true[:, :, 1]) * 0.5)
        xt_loss /= torch.clamp(torch.std(xt_true[:, :, :2]), min=0.1)

        latent_align_loss = F.mse_loss(xt_pred_hidden, xt_true_hidden)
        std_loss = F.mse_loss(xt_pred_hidden.std(dim=(0, 1)), xt_true_hidden.std(dim=(0, 1)))
        xt_loss += 0.2 * latent_align_loss
        xt_loss += 0.2 * std_loss

        spike_mask = (xt_true[:, :, 0] > 0.45).float()
        spike_penalty = ((xt_pred[:, :, 0] - xt_true[:, :, 0]) ** 2) * spike_mask
        if spike_mask.sum() > 0:
            spike_penalty = spike_penalty.sum() / spike_mask.sum()
        else:
            spike_penalty = torch.tensor(0.0, device=xt_true.device)

        small_mask = (xt_true[:, :, 0] < 0.035).float()
        small_penalty = ((xt_pred[:, :, 0] - xt_true[:, :, 0]) ** 2) * small_mask
        if small_mask.sum() > 0:
            small_penalty = small_penalty.sum() / small_mask.sum()
        else:
            small_penalty = torch.tensor(0.0, device=xt_true.device)

        xt_loss += 0.2 * spike_penalty
        xt_loss += 0.2 * small_penalty

        latent_loss = F.mse_loss(xt_pred, xt_true)
        cos_sim = F.cosine_similarity(xt_pred.flatten(1), xt_true.flatten(1), dim=1).mean()
        cosine_loss = 1 - cos_sim
        std_loss = F.mse_loss(xt_pred.std(dim=(0, 1)), xt_true.std(dim=(0, 1)))

        ##############################################################################epsilon
        epsilon_loss_std = torch.std(epsilon_true) + 1e-8

        # xt_loss = 0.7 * F.mse_loss(xt_pred_hidden, xt_true_hidden) + 0.3 * normalized_xt_loss

        # xt_loss = F.mse_loss(xt_pred_hidden, xt_true_hidden) / (xt_true.std() + 1e-8)

        # epsilon_f1 = F.smooth_l1_loss(epsilon_pred, epsilon_true)
        # spec_loss = fft_loss(epsilon_pred, epsilon_true)
        epsilon_loss = 0.5 * F.mse_loss(epsilon_pred, epsilon_true) + 0.5 * fft_loss(epsilon_pred, epsilon_true)
        normalized_epsilon_loss = epsilon_loss #/ epsilon_loss_std
        std_penalty = ((epsilon_pred.std() - 1.0) ** 2)
        normalized_epsilon_loss = normalized_epsilon_loss + 0.05 * std_penalty

        # total_loss = (1 - adaptive_weight) * xt_loss + adaptive_weight * normalized_epsilon_loss
        total_loss = normalized_epsilon_loss
        # total_loss = (
        #         0.6 * epsilon_loss +  # if you're still using ε supervision
        #         0.2 * latent_loss +  # structure alignment
        #         0.1 * cosine_loss +  # direction
        #         0.1 * std_loss  # scale
        # )
        return total_loss


def fft_loss(pred, target):
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))

def get_scheduled_k(epoch, total_epochs, K, min_k=0, max_k=None):
    """
    Computes the scheduled starting noise level for the diffusion process
    based on the current epoch using a cosine interpolation schedule.

    :param epoch: Current epoch number.
    :param total_epochs: Total number of training epochs.
    :param K: Total number of diffusion timesteps available.
    :param min_k: Minimum allowed starting timestep, default set to 50.
    :param max_k: Maximum allowed starting timestep, default K - 1.
    :return:
        - k_start: Scheduled starting noise level for the current epoch.
        - max_k: Maximum timestep for diffusion.

    Description:
        - Smoothly increases the difficulty using a half-cosine curve from min_k to max_k.
        - Early epochs use small k (less noise), later epochs use large k (more noise).
    """
    max_k = max_k if max_k is not None else K - 1
    progress = min(epoch / (total_epochs - 20), 1.0)
    k_start = int(min_k + (max_k - min_k) * (0.5 * (1 - np.cos(progress * np.pi))))
    # return k_start, max_k
    return K-1, K-1
def get_dynamic_loss_weights(epoch, total_epochs, start=0.8, end=0.2):
    """
    Computes a dynamic adaptive weight that controls the balance between
    noise prediction loss (epsilon) and consumption prediction loss (xt) during training.

    :param epoch: Current epoch number.
    :param total_epochs: Total number of training epochs.
    :return: Adaptive weight vallue for current epoch

    Description:
        - Early epochs focus more on noise (epsilon) prediction.
        - Later epochs shift focus towards clean signal (xt) prediction.
        - The transition follows a nonlinear schedule to balance learning stages.
    """
    progress = min(epoch / total_epochs, 1.0)
    # noise_weight = 0.7 - (0.5 * progress)
    # adaptive_weight = (0.7 - 0.3 * progress) * (1 - noise_weight)
    # return adaptive_weight #0.5
    # adaptive_weight = min(0.6, 0.6 * progress)
    return start + (end - start) * progress

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
    filename = f"{prefix}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved successfully at: {filepath}")

def extract_target_series(data, scaler, target_column):
    index = data.index
    target_values = scaler.inverse_transform(data)[:, scaler.feature_names_in_.tolist().index(target_column)]
    return pd.Series(target_values, index=index, name=target_column)

def plotting_preprocess_epsilon(epsilon_true_all, epsilon_pred_all):
    epsilon_true_avg = [eps.mean(axis=1) for eps in epsilon_true_all]
    epsilon_pred_avg = [eps.mean(axis=1) for eps in epsilon_pred_all]

    epsilon_true_flat = np.concatenate(epsilon_true_avg)
    epsilon_pred_flat = np.concatenate(epsilon_pred_avg)
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

def flatten_overlapping_windows_batched_last(window_list):
    output = [window[:, -1, :].unsqueeze(1) for window in window_list]
    return torch.cat(output, dim=1).squeeze(0)

def flatten_overlapping_windows_last(window_list):
    output = [window[-1:, :] for window in window_list]
    return torch.cat(output, dim=0)

def get_from_sequence(sequence, column="Consumption"):
    if column == "Consumption":
        return sequence[:, 0]
    if column == "Production":
        return sequence[:, 1]


