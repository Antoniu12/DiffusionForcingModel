import torch

from DiffusionBase.DF_Backbone import predict_start_from_noise
from DiffusionBase.df_training_v2 import forward_diffuse
from utils.utils import compute_sampling_step, flatten_overlapping_windows_batched_last

def sample_whole_test(model, test_data, alpha_bar,sequence_dim, feature_dim, hidden_dim, K, device, one_shot_denoising=True):
    predictions = []
    zt_prev = torch.zeros(
        (1, model.fc_project_seq_to_hidden.in_features, model.fc_project_seq_to_hidden.out_features),
        device=device
    )
    flag = True
    for test_seq in test_data:
        # xt_noisy = torch.randn((1, sequence_dim, feature_dim), device=device)
        # xt_noisy = model.fc_project_2_to_hidden(xt_noisy)

        x0 = test_seq.unsqueeze(0).to(device)
        xt_noisy = model.fc_project_2_to_hidden(x0)

        kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), K-1).to(device)
        xt_noisy = forward_diffuse(xt_noisy, kt, alpha_bar)
        # x0 = test_seq.unsqueeze(0).to(device)
        # x0 = model.fc_project_2_to_hidden(x0)
        # kt = torch.full((x0.shape[0], x0.shape[1]), K-1, dtype=torch.long, device=device)
        # xt_noisy = forward_diffuse(x0, kt, alpha_bar)
        #
        # x0_pred = predict_start_from_noise(xt_noisy, kt, noise, alpha_bar)
        if one_shot_denoising:
            x0_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
            print(f"eps std = {epsilon_pred.std().item():.4f}")
        else:
            for step in reversed(range(1, K-1, 1)):
                # if flag:
                    # print(f"step: {step}, xt_noisy: {xt_noisy}, xtnoisy.shape: {xt_noisy.shape}")
                kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), step, dtype=torch.long, device=device)
                _, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha_bar)
                # epsilon_pred = epsilon_pred.clamp(-0.15, 0.15)
                x0_est = predict_start_from_noise(xt_noisy, kt, epsilon_pred, alpha_bar)
                kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), step-1, dtype=torch.long, device=device)
                xt_noisy = forward_diffuse(x0_est, kt, alpha_bar)
                if flag:
                    print(f"Step {step}: eps std = {epsilon_pred.std().item():.4f}")
                    # print(f"step: {step}, xt-1: {xt_noisy}, xtnoisy.shape: {xt_noisy.shape}")
            kt = torch.full((xt_noisy.shape[0], xt_noisy.shape[1]), 0, dtype=torch.long, device=device)
            x0_pred = predict_start_from_noise(xt_noisy, kt, epsilon_pred, alpha_bar)
        zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()
        xt_pred = model.fc_project_xt_output(x0_pred)
        predictions.append(xt_pred.squeeze(0))
        flag = False
    return predictions
