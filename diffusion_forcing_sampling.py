def sample_from_diffusion(model, seq_len, alpha_bar, K, features=7, T=200):
    model.eval()
    with torch.no_grad():
        xt = torch.randn(1, seq_len, features).to(next(model.parameters()).device)
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features,
                               model.fc_project_seq_to_hidden.out_features)).to(xt.device)

        for step in reversed(range(T)):
            k = torch.full((xt.shape[0], xt.shape[1]), step, device=xt.device)
            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt, k, alpha_bar)
            xt = model.fc_project_xt_output(xt_pred)

            # Optionally you can add noise or blend in some randomness here
            zt_prev = zt_updated

        return xt.squeeze(0).cpu().numpy()
