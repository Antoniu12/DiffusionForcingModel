import os
from datetime import datetime

import matplotlib
import numpy as np

from DiffusionBase.DF_Backbone_NextToken import DFBackbone_NextToken
from DiffusionBase.df_training_next_token import train_next_token_diffusion, rolling_next_token_prediction, \
    predict_with_uncertainty_next_token, predict_with_random_last_noise, \
    autoregressive_forecast

# from DiffusionBase.diffusion_forcing_sampling import sample_whole_test, generate_predictions, next_hour_preddiction

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from utils.Logger import Logger
from utils.utils import compute_crps, save_model, flatten_overlapping_windows, \
    get_from_sequence
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors
from utils import utils
from DiffusionBase.DF_Backbone import DFBackbone  # , pretrain_layers
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty
from plots import plot_test_predictions, plot_predictions_with_uncertainty

#sa nu incerc pe H13!!!!
file_path = './training sets/H20_Wh.csv'
save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
os.makedirs(save_path, exist_ok=True)
logger = Logger(save_dir=save_path)

data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)

seq_length = 24
train_sequences = create_sequences(train_tensor, seq_length)
test_sequences = create_sequences(test_tensor, seq_length)
validation_sequences = create_sequences(validation_tensor, seq_length)
print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of validation sequences: {len(validation_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}")
input_dim = data_normalised.shape[1]
hidden_dim = input_dim
K = 1000
epochs = 100
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
# logger.log_model_config(model_name="DFBackbone", config_dict=model_config)
# device = 'cpu'
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)
#
forecast_window = 24
# stride = 1
#
#LOAD MODEL FOR PREDICTION
##############################################################################################################################
# context_seq = create_sequences(test_tensor[-forecast_window], seq_length)
# model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
# model.load_state_dict(torch.load("plots/With FULL K-1/df_model_2025-06-18_22-44-36.pth"))
#
# device = 'cpu'
# alpha = alpha.to(device)
# alpha_bar = alpha_bar.to(device)
# context_input = test_tensor[:-forecast_window]
# true_future = test_tensor[-forecast_window:]
#
# context_seq = create_sequences(context_input, seq_length)
#
#
# torch.cuda.empty_cache()
# # forecast = sample_from_diffusion_with_context(
# #     model, context_seq, seq_length, forecast_window, hidden_dim, alpha, alpha_bar, K
# # )
# context_seq = test_tensor[-seq_length:]
# true = test_tensor[-24:]
# true_sequence = create_sequences(test_tensor[-47:], seq_length)
#
# predictions = autoregressive_forecast_with_pred_as_next(
#     model=model,
#     context_seq=context_seq,      # shape: (24, input_dim)
#     alpha=alpha,
#     alpha_bar=alpha_bar,
#     K=K,
#     device=device,
#     steps=24,                     # How many hours to forecast into the future
#     )
# forecast = predictions
#
# forecast_production = forecast[:, 1].cpu().numpy()
# true_future = test_tensor[-24:, 1].cpu().numpy()
#
# plt.figure(figsize=(10, 5))
# plt.plot(true_future, label="True Production")
# plt.plot(forecast_production, label="Forecasted Production")
# plt.title("24-Step Autoregressive Diffusion Forecast")
# plt.xlabel("Time step")
# plt.ylabel("Normalized Production")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # Plot
# # plot_diffusion_forecast(forecast, ground_truth=true_future, column="Consumption")
##############################################################################################################################


#TRAIN MODEL
##############################################################################################################################
# train_tensor, val_tensor, test_tensor, scaler = preprocess_aep_dataset("./training sets/AEP_Wh.csv")
# input_dim = train_tensor.shape[1]
# train_sequences = create_sequences(train_tensor, seq_length)
# test_sequences = create_sequences(test_tensor, seq_length)
# validation_sequences = create_sequences(val_tensor, seq_length)






model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
model = model.to(device)
# pretrain_layers(model, train_sequences, total_epochs=20, device=device)
train_next_token_diffusion(model, train_sequences, validation_sequences, alpha, alpha_bar, K, epochs, device=device)
save_model(model, save_path, )

test_results = rolling_next_token_prediction(model, test_tensor, alpha, alpha_bar, K, device=device)
plot_test_predictions(test_results, scaler, save_path)

# test_results2 = predict_with_uncertainty_next_token(model, test_sequences, alpha, alpha_bar, K, device=device)

# crps_score = compute_crps(
#     np.array(test_results2["xt_true"]),
#     np.array(test_results2["xt_pred_mean"]),
#     np.array(test_results2["xt_pred_std"])
# )
#
# print(f"CRPS Score: {crps_score:.6f}")
#
# logger.save_final_metrics(crps_score)
#
# plot_predictions_with_uncertainty(test_results2, save_path)

#
# test_sequences = create_sequences(test_tensor, seq_length)
# predictions = sample_whole_test(model, test_sequences, alpha_bar, seq_length, input_dim, hidden_dim, K, device=device)
predictions = predict_with_random_last_noise(
    model=model,
    test_tensor=test_sequences,
    alpha=alpha,
    alpha_bar=alpha_bar,
    K=K,
    device=device
)

def flatten_overlapping_windows_preds(window_list):
    output = []
    for pred_tuple in window_list:
        arr = np.array(pred_tuple[1])
        if arr.ndim == 2:
            output.append(arr[-1])
        else:
            output.append(arr)
    return np.stack(output, axis=0)

def flatten_overlapping_windows_targets(window_list):
    output = []
    for arr in window_list:
        arr = np.array(arr)
        if arr.ndim == 2:
            output.append(arr[-1])
        else:
            output.append(arr)
    return np.stack(output, axis=0)

forecast = flatten_overlapping_windows_preds(predictions)
true = flatten_overlapping_windows_targets(test_sequences)

# Select correct feature index for "Production"
forecast_prod = forecast[:, 0]
true_prod = true[:, 0]

plt.figure(figsize=(10, 5))
plt.plot(true_prod, label="True Consumption")
plt.plot(forecast_prod, label="Forecasted Consumption")
plt.title("One-Step Forecast vs True Production")
plt.xlabel("Time step")
plt.ylabel("Normalized Production")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
##############################################################################################################################

#LSTM MODEL AND TRANSFORMER TRAINING
##############################################################################################################################
# file_path = './training sets'
#
# trained_models_root = os.path.join(".", "trained_models")
#
# lstm_model_dir = os.path.join(trained_models_root, "lstm", "model")
# lstm_scaler_dir = os.path.join(trained_models_root, "lstm", "scaler")
# transformer_model_dir = os.path.join(trained_models_root, "transformer", "model")
# transformer_scaler_dir = os.path.join(trained_models_root, "transformer", "scaler")
#
# os.makedirs(lstm_model_dir, exist_ok=True)
# os.makedirs(lstm_scaler_dir, exist_ok=True)
# os.makedirs(transformer_model_dir, exist_ok=True)
# os.makedirs(transformer_scaler_dir, exist_ok=True)
# #
# for i in range(1, 21):
#     file_name = f"H{i}_Wh.csv"
#     data_location = os.path.join(file_path, file_name)
#     print(f"Processing: {data_location}")
#     test_start_date = data_normalised.iloc[-len(test_tensor):].index[0]
#     test_end_date = data_normalised.iloc[-1].name
#
#     print(f"Test set covers from {test_start_date} to {test_end_date}")
#
#     data_normalised, scaler = load_and_preprocess_data(data_location, "1h")
#     train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)
#
#     input_dim = data_normalised.shape[1]
#
#     train_loader = DataLoader(SequenceDataset(train_tensor, seq_length), batch_size=32, shuffle=True)
#     val_loader = DataLoader(SequenceDataset(validation_tensor, seq_length), batch_size=32, shuffle=False)
#     test_loader = DataLoader(SequenceDataset(test_tensor, seq_length), batch_size=32, shuffle=False)
#
#     lstm_model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
#
#     transformer_model = TransformerRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
#
#     train_model(lstm_model, train_loader, val_loader, num_epochs=epochs, device=device)
#
#     torch.save(lstm_model.state_dict(), os.path.join(lstm_model_dir, f"H{i}.pth"))
#     with open(os.path.join(lstm_scaler_dir, f"H{i}.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
#
#     train_model(transformer_model, train_loader, val_loader, num_epochs=epochs, device=device)
#
#     torch.save(transformer_model.state_dict(), os.path.join(transformer_model_dir, f"H{i}.pth"))
#     with open(os.path.join(transformer_scaler_dir, f"H{i}.pkl"), "wb") as f:
#         pickle.dump(scaler, f)

# train_tensor, val_tensor, test_tensor, scaler = preprocess_aep_dataset("./training sets/AEP_Wh.csv")
# input_dim = train_tensor.shape[1]
# train_loader = DataLoader(SequenceDataset(train_tensor, seq_length), batch_size=32, shuffle=True)
# val_loader = DataLoader(SequenceDataset(val_tensor, seq_length), batch_size=32, shuffle=False)
# test_loader = DataLoader(SequenceDataset(test_tensor, seq_length), batch_size=32, shuffle=False)
#
# lstm_model = LSTMRegressor(input_dim, hidden_dim, input_dim)
# lstm_model.to(device)
# train_model(lstm_model, train_loader, val_loader, num_epochs=epochs, device=device)
# torch.save(lstm_model.state_dict(), os.path.join(lstm_model_dir, f"AEP.pth"))
# with open(os.path.join(lstm_scaler_dir, f"AEP.pkl"), "wb") as f:
#     pickle.dump(scaler, f)
#
# transformer_model = TransformerRegressor(input_dim, hidden_dim, input_dim)
# transformer_model.to(device)
# train_model(transformer_model, train_loader, val_loader, num_epochs=epochs, device=device)
# torch.save(transformer_model.state_dict(), os.path.join(transformer_model_dir, f"AEP.pth"))
# with open(os.path.join(transformer_scaler_dir, f"AEP.pkl"), "wb") as f:
#     pickle.dump(scaler, f)
##############################################################################################################################

# model = LSTMRegressor(input_dim=42, hidden_dim=512, output_dim=42)
# # forecast_day_from_model_aep(
# #     model=model,
# #     target_date=datetime(2018, 6, 21),
# #     csv_path="./training sets/AEP_Wh.csv",
# #     model_path="./trained_models/transformer/model/AEP.pth",
# #     scaler_path="./trained_models/transformer/scaler/AEP.pkl"
# # )
# forecast_day_from_model_h(
#     model=model,
#     target_date=datetime(2020, 12, 21),
#     csv_path="./training sets/H1_Wh.csv",
#     model_path="./trained_models/lstm/model/H1.pth",
#     scaler_path="./trained_models/lstm/scaler/H1.pkl"
# )
#


