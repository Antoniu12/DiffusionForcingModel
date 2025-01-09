# each token has a different noise leve k [0,K] (partial masking)
# Xt depends only on past noisy tokens
# implementation using RNN
# RNN maintains latents Zt
# Zt-1 and Xt noisy as input to the RNN to predict Xt unnoised and indirectly the noise epsilon via affine parameters

# uses a noise scheduler on a 2D MxT grid columns->t(time step), rows->m(noise level)
# initialize X1:t with white noise
# iterate down the grid row by row denoising left to right across the columns (by the last row m=0 tokens are clean)


import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import R2Score

import matplotlib.pyplot as plt
import random

def forward_diffuse(xt, kt, alpha):
    sqrt = torch.sqrt(alpha[kt])
    sqrt2 = torch.sqrt(1-alpha[kt])
    noise = torch.randn_like(xt)
    return sqrt * xt + sqrt2 * noise

def df_training(model, data, alpha, K, epochs):
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    loss_function = nn.MSELoss()
    r2_metric = R2Score()

    for epoch in range(epochs):
        epoch_loss = 0
        zt_prev = torch.zeros((1, 24 , (model.fc.in_features)))

        for trajectory in data:
            xt, label = trajectory
            #print(f"trajectory: {trajectory}")
            optimizer.zero_grad()
            trajectory_loss = 0
            t = 0
            #print(f"Iteration: {t}")
            t+=1
            #print(f"xt: {xt}")

            # print(f"zt_prev shape: {zt_prev.shape}")

            xt = xt.unsqueeze(0) # batch size
            xt = model.fc_project(xt)

            # print(f"xt shape: {xt.shape}")

            kt = torch.randint(0, K, (1,)).item()
            xt_noisy = forward_diffuse(xt, kt, alpha)
            # print(f"xt_noisy shape: {xt_noisy.shape}")

            sqrt = torch.sqrt(alpha[kt])
            sqrt2 = torch.sqrt(torch.clamp(1 - alpha[kt], min=1e-8))

            epsilon_true = (xt_noisy - sqrt * xt) / sqrt2
            epsilon_pred = model(zt_prev, xt_noisy)
            zt_prev_updated = epsilon_pred[..., :-16]
            zt_prev = zt_prev_updated.clone().detach()
            epsilon_pred = epsilon_pred[..., -16:]

            # print(epsilon_pred.shape)
            # print(epsilon_true.shape)
            trajectory_loss = loss_function(epsilon_pred, epsilon_true)

            trajectory_loss.backward()
            optimizer.step()
            epoch_loss += trajectory_loss

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm().item()}")
#r**2 cu cat mai aproape de 1 mai bine
#mape sau smape(mai ok)
#neaparat sa plotez
class DFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_project = nn.Linear(2, hidden_dim)
        self.RNN = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, zt_prev, xt_noisy):
        input = torch.cat([zt_prev, xt_noisy], dim=-1)
        # print(f"input shape: {input.shape}")
        output, _ = self.RNN(input)
        return self.fc(output)




