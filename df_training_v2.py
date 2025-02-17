# each token has a different noise leve k [0,K] (partial masking)
# Xt depends only on past noisy tokens
# implementation using RNN
# RNN maintains latents Zt
# Zt-1 and Xt noisy as input to the RNN to predict Xt unnoised and indirectly the noise epsilon via affine parameters

# uses a noise scheduler on a 2D MxT grid columns->t(time step), rows->m(noise level)
# initialize X1:t with white noise
# iterate down the grid row by row denoising left to right across the columns (by the last row m=0 tokens are clean)

#din training-uri am obersvat ca lr de 1e-4 ii mai bun decat 1e-8
#si rnn mai bun decat LSTM ???
#GRU mai bun decat LSTM si asemanator cu RNN
#lr 1e-1 bunicel dar instabil
#pe gpu training mai lent???
#lr 1e-2 foarte bun
#lr 1e-3 perfect
#AdamW > RAdam, Adam
#SGD slab cu lr 1e-3
#SmoothL1Loss genial pentru loss function

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import R2Score
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
import random

from plots import TrainingPlotter

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
def forward_diffuse(xt, kt, alpha):
    sqrt = torch.sqrt(alpha[kt])
    sqrt2 = torch.sqrt(1-alpha[kt])
    noise = torch.randn_like(xt)
    return sqrt * xt + sqrt2 * noise

def df_training(model, data, alpha, K, epochs):
    model#.to(device)
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    #loss_function = nn.MSELoss()
    loss_function = nn.SmoothL1Loss()

    r2_loss = R2Score()#.to(device)
    smape_loss = SymmetricMeanAbsolutePercentageError()#.to(device)

    plotter = TrainingPlotter()

    for epoch in range(epochs):
        epoch_loss, epoch_r2, epoch_smape = 0, 0, 0
        zt_prev = torch.zeros((1, 24 , model.fc.in_features))#.to(device)

        for trajectory in data:
            xt, _ = trajectory
            xt = xt#.to(device)

            #print(f"trajectory: {trajectory}")
            optimizer.zero_grad()
            #t = 0
            #print(f"Iteration: {t}")
            #t+=1
            #print(f"xt: {xt}")

            # print(f"zt_prev shape: {zt_prev.shape}")

            xt = xt.unsqueeze(0)
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
            r2_value = r2_loss(epsilon_pred.reshape(-1), epsilon_true.reshape(-1))
            smape_value = smape_loss(epsilon_pred, epsilon_true)

            trajectory_loss.backward()
            optimizer.step()

            epoch_loss += trajectory_loss.item()
            epoch_r2 += r2_value.item()
            epoch_smape += smape_value.item()

        plotter.update_metrics(epoch_loss / len(data), epoch_r2 / len(data), epoch_smape / len(data), epsilon_pred, epsilon_true)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm().item()}")
    plotter.plot_metrics()
class DFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_project = nn.Linear(2, hidden_dim)
        #self.RNN = nn.RNN(input_dim, hidden_dim, batch_first=True)
        #self.RNN = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)#, dropout=0.1)
        self.RNN = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, zt_prev, xt_noisy):
        input = torch.cat([zt_prev, xt_noisy], dim=-1)
        # print(f"input shape: {input.shape}")
        output, _ = self.RNN(input)
        return self.fc(output)

