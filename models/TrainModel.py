import copy
import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchmetrics.regression import MeanAbsolutePercentageError
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

def train_model(model, train_loader, val_loader, num_epochs=100, patience=10, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    best_val_loss = float('inf')
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)

def evaluate_test_dataset(
    model, test_loader, scaler, feature_index=0, device='cpu', clamp_zero=True
):
    model.eval()
    preds = []
    targets = []
    mape = MeanAbsolutePercentageError()
    smape = SymmetricMeanAbsolutePercentageError()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            out = outputs[:, -1, feature_index].cpu().numpy().flatten()
            tgt = y_batch[:, -1, feature_index].cpu().numpy().flatten()
            preds.append(out)
            targets.append(tgt)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if clamp_zero:
        preds = np.clip(preds, 0, 1)

    preds_denorm = []
    targets_denorm = []
    for p, t in zip(preds, targets):
        pred_vec = np.zeros(scaler.scale_.shape)
        true_vec = np.zeros(scaler.scale_.shape)
        pred_vec[feature_index] = p
        true_vec[feature_index] = t
        preds_denorm.append(scaler.inverse_transform([pred_vec])[0][feature_index])
        targets_denorm.append(scaler.inverse_transform([true_vec])[0][feature_index])

    preds_denorm = np.array(preds_denorm)
    targets_denorm = np.array(targets_denorm)

    mae = mean_absolute_error(targets_denorm, preds_denorm)
    mse = mean_squared_error(targets_denorm, preds_denorm)
    r2 = r2_score(targets_denorm, preds_denorm)
    smape_val = smape(torch.tensor(preds_denorm), torch.tensor(targets_denorm))
    mape_val = mape(torch.tensor(preds_denorm), torch.tensor(targets_denorm))
    smape_val = smape_val.item() * 100
    mape_val = mape_val.item() * 100
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "SMAPE": smape_val,
        "MAPE": mape_val,
        "preds": preds_denorm,
        "targets": targets_denorm,
    }
    return metrics
