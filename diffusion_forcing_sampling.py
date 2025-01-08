import torch
def df_sampling(model, alpha_bar, K, T, guidance_fn=None):
    """Sampling algorithm with optional guidance."""
    xt = torch.randn((T, model.fc.in_features))  # Initialize tokens as Gaussian noise
    zt = torch.zeros((1, 1, model.fc.in_features))  # Initialize latent state

    for t in range(T):
        for k in range(K - 1, -1, -1):  # Reverse through noise levels
            xt_noisy = xt.clone()
            epsilon_pred = model(zt, xt_noisy, k)  # Predict noise
            xt_denoised = (xt_noisy - torch.sqrt(1 - alpha_bar[k]) * epsilon_pred) / torch.sqrt(alpha_bar[k])

            if guidance_fn:  # Add guidance if provided
                xt_denoised += guidance_fn(xt_denoised, t)

            xt = xt_denoised
            zt = epsilon_pred.detach()  # Update latent state

    return xt
