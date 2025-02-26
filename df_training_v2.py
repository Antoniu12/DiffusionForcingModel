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
    # print(f"Before diffusion: xt shape = {xt.shape}")  #[1, 24, 16]

    sqrt1 = torch.sqrt(torch.clamp(alpha[kt], min=1e-8)).unsqueeze(-1)
    sqrt2 = torch.sqrt(torch.clamp(1 - alpha[kt], min=1e-8)).unsqueeze(-1)
    noise = torch.randn_like(xt)
    xt_noisy = sqrt1 * xt + sqrt2 * noise

    # print(f"After diffusion: xt_noisy shape = {xt_noisy.shape}")  #[1, 24, 16]

    return xt_noisy

def df_training(model, data, validation_data, alpha, K, epochs):
    #model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3, threshold=0.001)
    loss_function = nn.MSELoss()
    #loss_function = nn.SmoothL1Loss()

    plotter = TrainingPlotter()

    for epoch in range(epochs):
        epoch_loss, epoch_r2, epoch_smape = 0, 0, 0
        model.train()
        zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features , model.fc_project_seq_to_hidden.out_features))#.to(device)

        for trajectory in data:
            xt = trajectory
            #xt = xt.to(device)

            #print(f"trajectory: {trajectory}")
            optimizer.zero_grad()
            # t = 0
            # print(f"Iteration: {t}")
            # t+=1
            # print(f"xt: {xt}")

            # print(f"zt_prev shape: {zt_prev.shape}")
            # print(f"Before unsqeeze: xt shape = {xt.shape}")

            xt = xt.unsqueeze(0)
            # print(f"After unsqeeze: xt shape = {xt.shape}")

            #print(f"Before fc_project: xt shape = {xt.shape}")
            xt = model.fc_project_2_to_hidden(xt)
            #print(f"After fc_project: xt shape = {xt.shape}")

            # print(f"xt shape: {xt.shape}")

            kt = torch.randint(0, K, (xt.shape[0], xt.shape[1]))
            xt_noisy = forward_diffuse(xt, kt, alpha)
            # print(f"After forward_diffuse: xt_noisy shape = {xt_noisy.shape}")


            sqrt1 = torch.sqrt(torch.clamp(alpha[kt], min=1e-8)).unsqueeze(-1)
            sqrt2 = torch.sqrt(torch.clamp(1 - alpha[kt], min=1e-8)).unsqueeze(-1)

            epsilon_true = (xt_noisy - sqrt1 * xt) / sqrt2
            xt_pred, epsilon_pred, zt_updated = model(zt_prev, xt_noisy, kt, alpha)
            zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach() # de verificat !!!!


            # print(epsilon_pred.shape)
            # print(epsilon_true.shape)
            trajectory_loss = loss_function(epsilon_pred, epsilon_true)
            xt_loss = loss_function(xt_pred, xt)
            total_loss = trajectory_loss + xt_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            r2_loss = R2Score()  # .to(device)
            smape_loss = SymmetricMeanAbsolutePercentageError()  # .to(device)

            zt_prev = torch.zeros((1, model.fc_project_seq_to_hidden.in_features, model.fc_project_seq_to_hidden.out_features))  # .to(device)
            for trajectory in validation_data:
                xt = trajectory
                xt = xt.unsqueeze(0)
                xt = model.fc_project_2_to_hidden(xt)

                kt = torch.randint(0, K, (xt.shape[0], xt.shape[1]))
                xt_noisy = forward_diffuse(xt, kt, alpha)

                sqrt1 = torch.sqrt(torch.clamp(alpha[kt], min=1e-8)).unsqueeze(-1)
                sqrt2 = torch.sqrt(torch.clamp(1 - alpha[kt], min=1e-8)).unsqueeze(-1)
                epsilon_true = (xt_noisy - sqrt1 * xt) / sqrt2
                xt_pred, output, zt_updated = model(zt_prev, xt_noisy, kt, alpha)
                zt_prev = 0.7 * zt_prev + 0.3 * zt_updated.detach()

                trajectory_loss = loss_function(output, epsilon_true)
                xt_loss = loss_function(xt_pred, xt)
                total_loss = trajectory_loss + xt_loss

                r2_value = r2_loss(output.reshape(-1), epsilon_true.reshape(-1))
                smape_value = smape_loss(output, epsilon_true)

                val_loss += total_loss.item()
                epoch_r2 += r2_value.item()
                epoch_smape += smape_value.item()

        scheduler.step(val_loss / len(validation_data))

        plotter.update_metrics(val_loss / len(validation_data), epoch_r2 / len(validation_data), epoch_smape / len(validation_data), epsilon_pred, epsilon_true)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss/len(data):.4f}, Validation Loss: {val_loss / len(validation_data):.4f}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm().item()}")
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch + 1}, LR: {param_group['lr']}")
    plotter.plot_metrics()
class DFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_dim):
        super().__init__()
        self.fc_project_2_to_hidden = nn.Linear(7, hidden_dim)
        self.RNN = nn.RNN(hidden_dim * 2, hidden_dim, num_layers = 2, batch_first=True, dropout = 0.2)
        #self.RNN = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)#, dropout=0.1)
        #self.RNN = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.zt_transition = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_project_hidden_to_input = nn.Linear(hidden_dim, input_dim)
        self.fc_project_seq_to_hidden = nn.Linear(seq_dim, hidden_dim)

    def forward(self, zt_prev, xt_noisy, kt, alpha):
        # print(f"xt_noisy shape: {xt_noisy.shape}")
        # print(f"fc_project weight shape: {self.fc_project.weight.shape}")

        input = torch.cat([zt_prev, xt_noisy], dim=-1)
        # print(f"input shape: {input.shape}")
        output, _ = self.RNN(input)
        zt_updated, _ = self.zt_transition(output)
        epsilon_pred = self.fc_project_hidden_to_input(output)

        sqrt1 = torch.sqrt(torch.clamp(alpha[kt], min=1e-8)).unsqueeze(-1)
        sqrt2 = torch.sqrt(torch.clamp(1 - alpha[kt], min=1e-8)).unsqueeze(-1)
        # print(f"epsilon_pred shape: {epsilon_pred.shape}")
        # print(f"xt_noisy shape: {xt_noisy.shape}")
        # print(f"sqrt1 shape: {sqrt1.shape}")
        # print(f"sqrt2 shape: {sqrt2.shape}")

        xt_pred = (xt_noisy - sqrt2 * epsilon_pred) / sqrt1

        return xt_pred, epsilon_pred, zt_updated