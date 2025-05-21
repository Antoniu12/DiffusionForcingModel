import copy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import compute_crps


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.LSTM(x)
        return self.head(output)

def train_lstm(model, train_loader, val_loader, num_epochs=100, device='cpu', patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            target = y[:, :, [0, 1]]

            prediction = model(x)
            loss = mse_loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                target_val = y_val[:, :, [0, 1]]
                pred_val = model(x_val)
                val_loss += mse_loss_fn(pred_val, target_val).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)

def evaluate_lstm(model, test_loader, scaler, target_idxs=(0, 1), device='cpu'):
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_x_last = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_x_last.append(x[:, -1, :].cpu().numpy())

    pred = np.concatenate(all_preds, axis=0)
    y_test = np.concatenate(all_targets, axis=0)
    X_last = np.concatenate(all_x_last, axis=0)

    B, T, _ = pred.shape
    pred_inv = np.zeros_like(pred)
    y_inv = np.zeros_like(pred)

    for b in range(B):
        for t in range(T):
            x_base = X_last[b].copy()
            for i, idx in enumerate(target_idxs):
                x_base[idx] = pred[b, t, i]
            pred_inv[b, t, :] = scaler.inverse_transform([x_base])[0][list(target_idxs)]

            for i, idx in enumerate(target_idxs):
                x_base[idx] = y_test[b, t, i]
            y_inv[b, t, :] = scaler.inverse_transform([x_base])[0][list(target_idxs)]

    std_pred = np.std(pred_inv, axis=1, keepdims=True)
    std_pred = np.tile(std_pred, (1, T, 1))
    std_pred = np.clip(std_pred, 1e-3, None)

    crps = [compute_crps(y_inv[:, :, i].flatten(), pred_inv[:, :, i].flatten(), std_pred[:, :, i].flatten())
            for i in range(len(target_idxs))]

    return crps, y_inv, pred_inv

def plot_full_consumption_series(y_true, y_pred, stride=1, title="Full LSTM Forecast - Consumption"):
    B, T, _ = y_true.shape

    total_len = (B - 1) * stride + T
    true_series = np.zeros(total_len)
    pred_series = np.zeros(total_len)
    counts = np.zeros(total_len)

    for i in range(B):
        start = i * stride
        end = start + T

        true_series[start:end] += y_true[i, :, 0]
        pred_series[start:end] += y_pred[i, :, 0]
        counts[start:end] += 1

    counts[counts == 0] = 1

    true_avg = true_series / counts
    pred_avg = pred_series / counts

    plt.figure(figsize=(14, 5))
    plt.plot(true_avg, label="True Consumption", color='blue', linewidth=2)
    plt.plot(pred_avg, label="Predicted Consumption", color='orange', linewidth=2)
    plt.xlabel("Time (joined sequences)")
    plt.ylabel("Consumption")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

