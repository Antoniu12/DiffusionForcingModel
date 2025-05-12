import numpy as np
import torch
from torch import nn

from utils.utils import compute_crps


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

def train_lstm(model, train_loader, val_loader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch[:, :-1, :]
            y = batch[:, -1, 0:1]
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def train_lstm(model, X_train, y_train, X_val, y_val, epochs=30, lr=1e-3, device='cpu'):
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            val_loss = loss_fn(model(X_val), y_val).item()
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            model.train()

def split_sequences(sequences, target_col_idx):
    X = []
    y = []
    for seq in sequences:
        X.append(seq[:-1, :])  # toate timesteps -1
        y.append(seq[-1, target_col_idx])  # target doar la final
    return torch.stack(X), torch.tensor(y).unsqueeze(1)

def evaluate_lstm(model, X_test, y_test, scaler, target_idx, device='cpu'):
    model.eval()
    model = model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        pred = model(X_test).cpu().numpy()

    X_last = X_test[:, -1, :].cpu().numpy()

    # Create input copies for inverse transform
    X_true_full = X_last.copy()
    X_pred_full = X_last.copy()

    # Replace only the target column with true/predicted values
    X_true_full[:, target_idx] = y_test.cpu().numpy().flatten()
    X_pred_full[:, target_idx] = pred.flatten()

    # Inverse transform both
    y_test_inv = scaler.inverse_transform(X_true_full)[:, target_idx]
    pred_inv = scaler.inverse_transform(X_pred_full)[:, target_idx]

    std_pred = np.full_like(pred_inv, fill_value=np.std(pred_inv) * 0.1)

    crps = compute_crps(y_test_inv, pred_inv, std_pred)
    return crps, y_test_inv, pred_inv

