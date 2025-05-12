import os
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch

from DiffusionBase.diffusion_forcing_sampling import sample_from_diffusion_with_context
from models.Lstm import LSTMRegressor, split_sequences, train_lstm, evaluate_lstm

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from models.Arima import forecast_arima, forecast_prophet
from utils.Logger import Logger
from utils.utils import compute_crps, save_model, extract_target_series

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors, \
    plot_feature_correlation_heatmap
from utils import utils
from DiffusionBase.DF_Backbone import DFBackbone
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty
from plots import plot_test_predictions, plot_predictions_with_uncertainty, plot_diffusion_forecast

#sa nu incerc pe H13!!!!
file_path = './training sets/H4_Wh.csv'
save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
os.makedirs(save_path, exist_ok=True)
logger = Logger(save_dir=save_path)

data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
plot_feature_correlation_heatmap(data_normalised, save_path)
train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)

seq_length = 24
train_sequences = create_sequences(train_tensor, seq_length)
test_sequences = create_sequences(test_tensor, seq_length)
validation_sequences = create_sequences(validation_tensor, seq_length)
print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of validation sequences: {len(validation_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}")

input_dim = data_normalised.shape[1]
hidden_dim = 512
K = 1000
epochs = 60
betas = utils.cosine_beta_schedule(K)
alpha, alpha_bar = utils.get_alphas(betas)
loss_type = "spike_and_small_masked"
model_config = {
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'K': K,
    'epochs': epochs,
    'loss_type': "spike_and_small_masked",
}
logger.log_model_config(model_name="DFBackbone", config_dict=model_config)

alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

context_hours = 48
steps_ahead = 24
stride = 1

# Input: context sequence (from preprocessed/scaled data)
# context_seq = create_sequences(test_tensor[-context_hours:], seq_length)
# context_seq = torch.stack(context_seq).to(device)
# model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
# model.load_state_dict(torch.load("plots/benchmark/df_model_2025-05-04_17-59-06.pt"))
# model.to(device)

# forecast = sample_from_diffusion_with_context(model, context_seq, steps_ahead, hidden_dim, alpha, alpha_bar, K)
# context_slice = context_seq[0, :, 0].cpu().numpy()
# forecast_slice = forecast[:, 0]
#
# plot_diffusion_forecast(context_slice, forecast_slice)

model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
model = model.to(device)

df_training(model, train_sequences, validation_sequences, alpha_bar, K, epochs,
            scaler, loss_type, device=device, save_path=save_path)
save_model(model, save_path, )

test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
plot_test_predictions(test_results, scaler, save_path)

test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)

crps_score = compute_crps(
    np.array(test_results2["xt_true"]),
    np.array(test_results2["xt_pred_mean"]),
    np.array(test_results2["xt_pred_std"])
)

print(f"CRPS Score: {crps_score:.6f}")

logger.save_final_metrics(crps_score)
plot_predictions_with_uncertainty(test_results2, save_path)


target_col = " Consumption(Wh)"
target_idx = list(scaler.feature_names_in_).index(target_col)
test_np = test_tensor.numpy()
test_inv = scaler.inverse_transform(test_np)
test_target = test_inv[:, target_idx]

full_np = torch.cat([train_tensor, validation_tensor], dim=0).numpy()
full_inv = scaler.inverse_transform(full_np)
target_idx = list(scaler.feature_names_in_).index(" Consumption(Wh)")
train_target = full_inv[:, target_idx]
test_np = test_tensor.numpy()
test_inv = scaler.inverse_transform(test_np)
test_target = test_inv[:, target_idx]
arima_mean, arima_std = forecast_arima(train_target, steps=len(test_target))
crps_arima = compute_crps(test_target, arima_mean, arima_std)

trainval_tensor = torch.cat([train_tensor, validation_tensor], dim=0)
trainval_np = trainval_tensor.numpy()
trainval_inv = scaler.inverse_transform(trainval_np)
target_idx = list(scaler.feature_names_in_).index(" Consumption(Wh)")
target_vals = trainval_inv[:, target_idx]
trainval_index = data_normalised.index[:len(target_vals)]
train_target_series = pd.Series(data=target_vals, index=trainval_index, name="y")
prophet_mean, prophet_std = forecast_prophet(train_target_series, steps=len(test_target), freq="H")
crps_prophet = compute_crps(test_target, prophet_mean, prophet_std)
print(f"[Consumption] ARIMA CRPS: {crps_arima:.4f}, Prophet CRPS: {crps_prophet:.4f}")
model = LSTMRegressor(input_dim=input_dim).to(device)
target_idx = list(scaler.feature_names_in_).index(" Consumption(Wh)")
X_train, y_train = split_sequences(train_sequences, target_idx)
X_val, y_val = split_sequences(validation_sequences, target_idx)
X_test, y_test = split_sequences(test_sequences, target_idx)
train_lstm(model, X_train, y_train, X_val, y_val, device=device)
crps_lstm, true, pred = evaluate_lstm(model, X_test, y_test, scaler, target_idx)
print("LSTM CRPS:", crps_lstm)

plt.figure(figsize=(12, 6))

plt.plot(test_target, label="Ground truth", color='black', linewidth=2.5)
plt.plot(arima_mean, label="ARIMA", color='darkorange', linewidth=2)
plt.plot(prophet_mean, label="Prophet", color='forestgreen', linewidth=2)
plt.plot(pred, label="LSTM", color='crimson', linewidth=2)

plt.legend(fontsize=12)
plt.title("Predictions vs Ground Truth", fontsize=14)
plt.xlabel("Timestep", fontsize=12)
plt.ylabel("Consumption (Wh)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(save_path, "predictions_vs_ground_truth.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")

try:
    from PIL import Image
    Image.open(plot_path).show()
except ImportError:
    print("Install Pillow (`pip install pillow`) to auto-open the saved plot.")
