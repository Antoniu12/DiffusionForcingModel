import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def df_sampling(model, K, alpha, T, M, guided=False, guidance_cost=None):
    """
    Implements Diffusion Forcing Sampling with optional guidance.

    :param model: Trained DFModel
    :param K: Noise scheduling matrix of shape [M, T]
    :param alpha: Precomputed noise schedule
    :param T: Sequence length
    :param M: Number of noise levels
    :param guided: Whether to use guidance
    :param guidance_cost: Function for computing the gradient of the guidance cost
    :return: Generated sequence x_1:T
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Step 2: Initialize x_1:T with white noise
    # Ensure input shape is correct for the RNN model
    zt_prev = torch.zeros((1, T, 16))  # Shape: (1, T, hidden_dim=16)
    xt_noisy = torch.randn((1, T, 16))  # Shape: (1, T, 16)

    # Iterate through noise levels
    for m in range(M - 1, -1, -1):
        for t in range(T):
            with torch.no_grad():
                prev_state = xt_noisy[:, max(0, t - 1):t, :]
                current_state = xt_noisy[:, t:t + 1, :]

                if prev_state.shape[1] == 0:
                    prev_state = torch.zeros_like(current_state)

                input_tensor = torch.cat([zt_prev[:, t:t + 1, :], current_state], dim=-1)  # Shape (1, 1, 32)
                z_t_new = model(prev_state, input_tensor)  # Correct shape now

            # Sample noise level
            k = K[m, t]
            w = torch.randn_like(xt_noisy[:, t:t + 1])  # Gaussian noise

            sqrt_alpha_k = torch.sqrt(alpha[k])
            sqrt_one_minus_alpha_k = torch.sqrt(1 - alpha[k])

            epsilon_theta = model(z_t_new, xt_noisy[:, t:t + 1])
            x_t_new = (1 / sqrt_alpha_k) * (xt_noisy[:, t:t + 1] - (
                        1 - alpha[k]) / sqrt_one_minus_alpha_k * epsilon_theta) + sqrt_one_minus_alpha_k * w

            # Update z_t
            xt_noisy[:, t:t + 1] = x_t_new

        # Step 10: Apply guidance if enabled
        if guided and guidance_cost is not None:
            grad = torch.autograd.grad(guidance_cost(x_t), x_t, retain_graph=True)[0]
            x_t += grad * 0.1  # Small step in gradient direction

    return x_t.cpu().numpy()

# Prediction function
def predict(model, test_sequences, K, alpha, T, M):
    """
    Use DF Sampling to generate predictions on the test set.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for seq, _ in test_sequences:
            seq = seq.unsqueeze(0)  # Add batch dimension
            pred = df_sampling(model, K, alpha, T, M, guided=False)
            predictions.append(pred)

    return np.array(predictions)
