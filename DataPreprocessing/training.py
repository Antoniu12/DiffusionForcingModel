import os
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DiffusionBase.DF_Backbone_UNet import DFBackboneUNet
from DiffusionBase.diffusion_forcing_sampling import sample_whole_test
from models.Lstm import LSTMRegressor, train_lstm, evaluate_lstm, plot_full_consumption_series
from utils.Sequence_Dataset import SequenceDataset

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from models.Arima import forecast_arima, forecast_prophet
from utils.Logger import Logger
from utils.utils import compute_crps, save_model, extract_target_series, flatten_overlapping_windows, \
    flatten_overlapping_windows_last, get_from_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors, \
    plot_feature_correlation_heatmap
from utils import utils
from DiffusionBase.DF_Backbone import DFBackbone, pretrain_layers
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty
from plots import plot_test_predictions, plot_predictions_with_uncertainty, plot_diffusion_forecast

#sa nu incerc pe H13!!!!
file_path = './training sets/H4_Wh.csv'
save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
os.makedirs(save_path, exist_ok=True)
logger = Logger(save_dir=save_path)

data_normalised, scaler = load_and_preprocess_data(file_path, "D")
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
epochs = 120
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
# device = 'cpu'
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

forecast_window = 24
stride = 1

#LOAD MODEL FOR PREDICTION
##############################################################################################################################
# # context_seq = create_sequences(test_tensor[-forecast_window], seq_length)
# model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
# model.load_state_dict(torch.load("plots/2025-05-16_17-25/df_model_2025-05-16_17-27-53.pt"))
# model.to(device)
#
# # context_input = test_tensor[:-forecast_window]
# # true_future = test_tensor[-forecast_window:]
# #
# # context_seq = create_sequences(context_input, seq_length)
#
# #
# torch.cuda.empty_cache()
# # forecast = sample_from_diffusion_with_context(
# #     model, context_seq, seq_length, forecast_window, hidden_dim, alpha, alpha_bar, K
# # )
# test_sequences = create_sequences(test_tensor[-forecast_window:], seq_length)
# predictions = sample_whole_test(model, test_sequences, alpha_bar, seq_length, hidden_dim, K, device)
#
# true = flatten_overlapping_windows(test_sequences)
# forecast = flatten_overlapping_windows(predictions)
#
# forecast = get_from_sequence(forecast.detach(), column="Consumption").cpu().numpy()
# true_future = get_from_sequence(true, column="Consumption").cpu().numpy()
#
# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(true_future, label="True Consumption")
# plt.plot(forecast, label="Forecasted Consumption")
# plt.title("One-Step Forecast vs True Consumption")
# plt.xlabel("Time step")
# plt.ylabel("Normalized Consumption")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # Plot
# # plot_diffusion_forecast(forecast, ground_truth=true_future, column="Consumption")
##############################################################################################################################


#TRAIN MODEL
##############################################################################################################################
model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
model = model.to(device)
pretrain_layers(model, train_sequences, total_epochs=60, device=device)
df_training(model, train_sequences, validation_sequences, alpha_bar, K, epochs,
            scaler, loss_type, device=device, save_path=save_path, one_shot_training=False)
save_model(model, save_path, )

test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
plot_test_predictions(test_results, scaler, save_path)
# alpha_bar = alpha_bar.flip(0).contiguous()

test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)

crps_score = compute_crps(
    np.array(test_results2["xt_true"]),
    np.array(test_results2["xt_pred_mean"]),
    np.array(test_results2["xt_pred_std"])
)

print(f"CRPS Score: {crps_score:.6f}")

logger.save_final_metrics(crps_score)
#
plot_predictions_with_uncertainty(test_results2, save_path)



# model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
# model.load_state_dict(torch.load("plots/2025-05-20_16-54/df_model_2025-05-20_16-58-23.pt"))
# model.to(device)
# torch.cuda.empty_cache()
# test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
# plot_test_predictions(test_results, scaler, save_path)
# forecast = sample_from_diffusion_with_context(
#     model, context_seq, seq_length, forecast_window, hidden_dim, alpha, alpha_bar, K
# )
test_sequences = create_sequences(test_tensor[-forecast_window:], seq_length)
predictions = sample_whole_test(model, test_sequences, alpha_bar, seq_length, input_dim, hidden_dim, K, device=device)
# test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
# plot_predictions_with_uncertainty(test_results2, save_path)

true = flatten_overlapping_windows(test_sequences)
forecast = flatten_overlapping_windows(predictions)

forecast = get_from_sequence(forecast.detach(), column="Consumption").cpu().numpy()
true_future = get_from_sequence(true, column="Consumption").cpu().numpy()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(true_future, label="True Consumption")
plt.plot(forecast, label="Forecasted Consumption")
plt.title("One-Step Forecast vs True Consumption")
plt.xlabel("Time step")
plt.ylabel("Normalized Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
##############################################################################################################################

#ARIMA MODEL
##############################################################################################################################
# target_col = " Consumption(Wh)"
# target_idx = list(scaler.feature_names_in_).index(target_col)
#
# full_np = torch.cat([train_tensor, validation_tensor], dim=0).numpy()
# full_inv = scaler.inverse_transform(full_np)
# test_inv = scaler.inverse_transform(test_tensor.numpy())
#
# train_target = full_inv[:, target_idx]
# test_target = test_inv[:, target_idx]
#
# arima_mean, arima_std = forecast_arima(train_target, steps=len(test_target))
# crps_arima = compute_crps(test_target, arima_mean, arima_std)
# print(f"[Consumption] ARIMA CRPS: {crps_arima:.4f}")
##############################################################################################################################

#PROPHET MODEL
##############################################################################################################################
# trainval_index = data_normalised.index[:len(train_target)]
# train_target_series = pd.Series(data=train_target, index=trainval_index, name="y")
#
# prophet_mean, prophet_std = forecast_prophet(train_target_series, steps=len(test_target), freq="H")
# crps_prophet = compute_crps(test_target, prophet_mean, prophet_std)
# print(f"[Consumption] Prophet CRPS: {crps_prophet:.4f}")
##############################################################################################################################

#LSTM MODEL
##############################################################################################################################
# train_loader = DataLoader(SequenceDataset(train_tensor, 24), batch_size=32, shuffle=True)
# val_loader = DataLoader(SequenceDataset(validation_tensor, 24), batch_size=32, shuffle=False)
# test_loader = DataLoader(SequenceDataset(test_tensor[:-forecast_window], 24), batch_size=32, shuffle=False)
# lstm_model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
#
# train_lstm(lstm_model, train_loader, val_loader, num_epochs=100, device=device)
# crps, y_inv, pred_inv = evaluate_lstm(lstm_model, test_loader, scaler, device=device)
# print("LSTM CRPS: ", crps)
# plot_full_consumption_series(y_inv, pred_inv, stride=1)



# B, T, _ = pred_inv.shape
# stride = 1
# total_len = (B - 1) * stride + T
# lstm_series = np.zeros(total_len)
# count_series = np.zeros(total_len)
#
# for i in range(B):
#     start = i * stride
#     end = start + T
#     lstm_series[start:end] += pred_inv[i, :, 0]
#     count_series[start:end] += 1
#
# count_series[count_series == 0] = 1
# lstm_avg = lstm_series / count_series
#
# # Match length to test_target
# lstm_avg = lstm_avg[:len(test_target)]
#
# # Plot comparison
# plt.figure(figsize=(14, 6))
# plt.plot(test_target, label="Ground Truth", color='black', linewidth=2.5)
# plt.plot(arima_mean, label="ARIMA", color='orange')
# plt.plot(prophet_mean, label="Prophet", color='green')
# plt.plot(lstm_avg, label="LSTM", color='crimson')
#
# plt.legend(fontsize=12)
# plt.title("Forecast Comparison - Consumption", fontsize=14)
# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Consumption (Wh)", fontsize=12)
# plt.grid(True)
# plt.tight_layout()
#
# plot_path = os.path.join(save_path, "final_forecast_comparison.png")
# plt.savefig(plot_path, dpi=300)
# print(f"Forecast comparison plot saved to: {plot_path}")
##############################################################################################################################

# plt.figure(figsize=(12, 6))
#
# plt.plot(test_target, label="Ground truth", color='black', linewidth=2.5)
# plt.plot(arima_mean, label="ARIMA", color='darkorange', linewidth=2)
# plt.plot(prophet_mean, label="Prophet", color='forestgreen', linewidth=2)
# plt.plot(pred, label="LSTM", color='crimson', linewidth=2)
#
# plt.legend(fontsize=12)
# plt.title("Predictions vs Ground Truth", fontsize=14)
# plt.xlabel("Timestep", fontsize=12)
# plt.ylabel("Consumption (Wh)", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
#
# plot_path = os.path.join(save_path, "predictions_vs_ground_truth.png")
# plt.savefig(plot_path, dpi=300)
# print(f"Plot saved to: {plot_path}")
#
# try:
#     from PIL import Image
#     Image.open(plot_path).show()
# except ImportError:
#     print("Install Pillow (`pip install pillow`) to auto-open the saved plot.")
