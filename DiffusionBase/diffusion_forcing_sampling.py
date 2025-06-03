import torch

from DiffusionBase.DF_Backbone import predict_start_from_noise
from DiffusionBase.df_training_v2 import forward_diffuse
from utils.utils import compute_sampling_step, flatten_overlapping_windows_batched_last

def sample_whole_test(model, test_data, alpha, alpha_bar, sequence_dim, feature_dim, hidden_dim, K, device, one_shot_denoising=False):
    predictions = []
    zt_prev = torch.zeros((1, sequence_dim, hidden_dim),device=device)
    flag = True
    for test_seq in test_data:
        x0 = test_seq.unsqueeze(0).to(device)
        xt_noisy = model.encoder(x0)

        kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), K-1).to(device)
        # xt_noisy = forward_diffuse(xt_noisy, kt, alpha_bar)

        if one_shot_denoising:
            x0_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            print(f"eps std = {epsilon_pred.std().item():.4f}")
        else:
            for step in reversed(range(1, 600, 100)):

                kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), step, dtype=torch.long, device=device)
                _, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
                # kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), step-1, dtype=torch.long, device=device)
                x_k_prev = compute_sampling_step(xt_noisy, epsilon_pred, kt, alpha, alpha_bar)
                x0_est = model.encoder(x0_est)
                xt_noisy = forward_diffuse(x0_est, kt, alpha_bar)
                if flag:
                    print(f"Step {step}: eps std = {epsilon_pred.std().item():.4f}")
                    print(f"step: {step}, xt-1: {xt_noisy}, xtnoisy.shape: {xt_noisy.shape}")
            kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), 0, dtype=torch.long, device=device)
            x0_pred = predict_start_from_noise(xt_noisy, kt, epsilon_pred, alpha_bar)
        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
        predictions.append(x0_pred.squeeze(0))
        flag = False
    return predictions

def generate_predictions(model, context_seq, alpha, alpha_bar, sequence_dim, feature_dim, hidden_dim, K, steps=1, device="cpu"):
    predictions = []
    zt_prev = torch.zeros((1, sequence_dim, hidden_dim), device=device)
    for test_sequence in context_seq:
        x_true = test_sequence.unsqueeze(0).to(device)
        x_true_hidden = model.encoder(x_true)

        kt = torch.full((x_true_hidden.shape[0], x_true_hidden.shape[1]), 10).to(device)
        xt_noisy = forward_diffuse(x_true_hidden, kt, alpha_bar)
        _, _, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
    xt = context_seq[-1].unsqueeze(0).to(device)
    xt_rand = torch.rand((1, sequence_dim, feature_dim), dtype=torch.float, device=device)
    xt = torch.cat([xt[:, 1:, :], xt_rand[:, -1:, :]], dim=1)

    for _ in range(steps):
        xt_noisy = model.encoder(xt)
        kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), 100).to(device)
        xt_noisy = forward_diffuse(xt_noisy, kt, alpha_bar)
        for k in reversed(range(0, 101, 100)):
            kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), k).to(device)
            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            # xt_noisy = compute_sampling_step(xt_noisy,epsilon_pred, kt, alpha, alpha_bar)
            print(f"k: {k}")
        # input_xt = torch.cat([xt_pred, zt_prev], dim=-1)
        # x0 = model.xt_head(input_xt)
        # x0 = model.fc_project_xt_output(x0)
        print(xt_pred)
        predictions.append(xt_pred[:, -1:, ].squeeze(0))
        xt = torch.cat([xt[:, 1:, :], xt_pred[:, -1:, :].detach()], dim=1)

        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
    return predictions

