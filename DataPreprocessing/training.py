import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors
from utils import utils
from DiffusionBase.DF_Backbone import DFBackbone
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty


from plots import plot_test_predictions, plot_predictions_with_uncertainty
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sa nu incerc pe H13!!!!
file_path = './training sets/H4_Wh.csv'
data_normalised, scaler = load_and_preprocess_data(file_path, "D")
train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)

seq_length = 24
train_sequences = create_sequences(train_tensor, seq_length)
test_sequences = create_sequences(test_tensor, seq_length)
validation_sequences = create_sequences(validation_tensor, seq_length)
print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of validation sequences: {len(validation_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}")

input_dim = 32
hidden_dim = 128
K = 1000
epochs = 50
betas = utils.cosine_beta_schedule(K)
alpha, alpha_bar = utils.get_alphas(betas)

model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
df_training(model, train_sequences, validation_sequences, alpha_bar, K, epochs, loss="mse+l1")
test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler)
plot_test_predictions(test_results, scaler)
test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler)
plot_predictions_with_uncertainty(test_results2)
