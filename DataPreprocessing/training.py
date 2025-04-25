import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import utils
from DiffusionBase.DF_Backbone import DFBackbone
from DiffusionBase.df_training_v2 import df_training, predict, predict_with_uncertainty
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from plots import plot_test_predictions, plot_predictions_with_uncertainty

#confidence interval!!!
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sa nu incerc pe H13!!!!

#seasonal decomposition !!!
#vanishing gradient/explosion gradient !!!
file_path = './training sets/H8_Wh.csv'
data = pd.read_csv(file_path)

data = data[['date', ' Consumption(Wh)']]

data['date'] = pd.to_datetime(data['date'])

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

data.set_index('date', inplace=True)
data = data.resample('D').sum() #!!!fara mean()
decomposition = seasonal_decompose(data[' Consumption(Wh)'], model='additive', period=24)

data['trend'] = decomposition.trend.fillna(0).cumsum()
data['seasonal'] = decomposition.seasonal.fillna(0).cumsum()
data['residual'] = decomposition.resid.fillna(0).cumsum()
#data leakage
#trend residual din residuals

data_normalised = data.copy()
scaler = MinMaxScaler()
data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']] = scaler.fit_transform(
    data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']]
)
data_normalised = data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']]

aux_data, test_data = train_test_split(data_normalised, test_size=0.1, shuffle = False)
train_data, validation_data = train_test_split(aux_data, test_size=0.2, shuffle = False)

train_tensor = torch.tensor(train_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)
validation_tensor = torch.tensor(validation_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)
test_tensor = torch.tensor(test_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)
# print("test_data: ", test_tensor[:, 3])
def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        seq = data_tensor[i:i+seq_length]
        sequences.append(seq)
    return sequences

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

model = DFBackbone(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)#.to(device)
df_training(model, train_sequences, validation_sequences, alpha, alpha_bar, K, epochs, loss="dinamic")
test_results = predict(model, test_sequences, alpha, alpha_bar, K, scaler)
plot_test_predictions(test_results, scaler)
test_results2 = predict_with_uncertainty(model, test_sequences, alpha, alpha_bar, K, scaler)
plot_predictions_with_uncertainty(test_results2)
