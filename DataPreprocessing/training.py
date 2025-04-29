import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils.utils import compute_crps, save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors, \
    plot_feature_correlation_heatmap
from utils import utils
from DiffusionBase.DF_Backbone import DFBackbone
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty
from plots import plot_test_predictions, plot_predictions_with_uncertainty

#sa nu incerc pe H13!!!!
file_path = './training sets/H4_Wh.csv'
save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))

data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
plot_feature_correlation_heatmap(data_normalised)
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

alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
model = model.to(device)

df_training(model, train_sequences, validation_sequences, alpha_bar, K, epochs,
            scaler, loss_type="spike_and_small_masked", device=device)
save_model(model)

test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
plot_test_predictions(test_results, scaler)

test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler, device=device)
crps_score = compute_crps(
    np.array(test_results2["xt_true"]),
    np.array(test_results2["xt_pred_mean"]),
    np.array(test_results2["xt_pred_std"])
)

print(f"CRPS Score: {crps_score:.6f}")
plot_predictions_with_uncertainty(test_results2)

