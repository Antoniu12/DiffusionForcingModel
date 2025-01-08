import torch
import torch.nn as nn
import torch.optim as optim
import random

def forward_diffuse(xt, kt, alpha):
    sqrt = torch.sqrt(alpha[kt])
    sqrt2 = torch.sqrt(1-alpha[kt])
    noise = torch.randn_like(xt)
    return sqrt * xt + sqrt2 * noise

class DFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.RNN = nn.RNN(input_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, zt_prev, xt_noisy):
        input = torch.cat([zt_prev, xt_noisy], dim= -1)
        output, _ = self.RNN(input)
        return self.fc(output)

def df_training(model, data, alpha, K, epochs):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for trajectory in data:
            optimizer.zero_grad()
            zt_prev = torch.zeros((1, 1, model.fc.in_features))
            trajectory_loss = 0

            for t, xt in enumerate(trajectory):
                kt = random.randint(0, K-1)
                xt_noisy = forward_diffuse(xt, kt, alpha)
                sqrt = torch.sqrt(alpha[kt])
                sqrt2 = torch.sqrt(1 - alpha[kt])
                epsilon_true = (xt_noisy - sqrt * xt_noisy) / sqrt2
                epsilon_true = epsilon_true.unsqueeze(0).unsqueeze(-1)  # Shape: [1, 24, 1]

                xt_noisy = xt_noisy.unsqueeze(0).unsqueeze(-1)
                zt_prev = zt_prev.repeat(1, xt_noisy.size(1), 1)
                xt_noisy_projected = torch.nn.Linear(1, 16)(xt_noisy)
                print(f"Iteration: t={t}")
                print(f"zt_prev shape: {zt_prev.shape}")
                print(f"xt_noisy shape: {xt_noisy.shape}")

                print(f"epsilon_true shape: {epsilon_true.shape}")

                epsilon_pred = model(zt_prev, xt_noisy_projected)
                epsilon_true = epsilon_true.repeat(1, 1, epsilon_pred.size(-1))  # Shape: [1, 24, 16]
                zt_prev = epsilon_pred.clone().detach()
                epsilon_true = epsilon_true.expand_as(epsilon_pred)
                print(f"epsilon_pred shape: {epsilon_pred.shape}")
                trajectory_loss = loss_function(epsilon_pred, epsilon_true)

            trajectory_loss.backward()
            optimizer.step()
            epoch_loss += trajectory_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


