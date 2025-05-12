import torch
from DiffusionBase.df_training_v2 import forward_diffuse
from utils.utils import compute_sampling_step


def sample_from_diffusion_with_context(model, context_seq, steps_ahead, hidden_dim,
                                       alpha, alpha_bar, K, stride=1, num_steps=2):
    model.eval()
    device = next(model.parameters()).device
    context_seq = context_seq.clone().detach().to(device)
    predictions = []

    # Step 1: Encode context into zt_prev using k=0 (no noise)
    zt_prev = torch.zeros((1,
                           model.fc_project_seq_to_hidden.in_features,
                           model.fc_project_seq_to_hidden.out_features),
                          device=device)

    for trajectory in context_seq:
        x0 = trajectory.unsqueeze(0).to(device)
        x0 = model.fc_project_2_to_hidden(x0)
        k0 = torch.zeros((1, x0.shape[1]), dtype=torch.long, device=device)
        _, _, zt_updated = model(zt_prev, x0, k0, alpha_bar)
        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()

    # Step 2: Initialize xt with Gaussian noise
    xt = torch.randn(1, steps_ahead, hidden_dim, device=device)

    # Step 3: Define denoising schedule (e.g. 100 uniform steps from K-1 to 0)
    timesteps = torch.linspace(K - 1, 0, steps=num_steps, dtype=torch.long)

    for i in range(len(timesteps) - 1):
        k = timesteps[i].item()
        k_next = timesteps[i + 1].item()
        kt = torch.full((xt.shape[0], xt.shape[1]), int(k), device=device, dtype=torch.long)

        # Step 4: Model predicts ε (noise)
        _, epsilon_pred, zt_updated = model(zt_prev, xt, kt, alpha_bar)
        epsilon_pred = epsilon_pred.clamp(-2.0, 2.0)

        # Step 5: Update xt ← xt-1 using your DDPM-based compute_sampling_step
        xt = compute_sampling_step(xt, epsilon_pred, kt, alpha, alpha_bar, add_noise=(k_next > 0), eta=0.0)

        # Step 6: Update latent
        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()

    # Step 7: Final projection to observable space
    xt = model.fc_project_xt_output(xt).clamp(0, 1)

    predictions.append(xt)

    forecast = torch.cat(predictions, dim=1).squeeze(0).detach().cpu().numpy()
    return forecast
