import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from df_training_v2 import DFModel, df_training
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

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
data = data.resample('1h').sum() #!!!fara mean()
decomposition = seasonal_decompose(data[' Consumption(Wh)'], model='additive', period=24)

data['trend'] = decomposition.trend
data['seasonal'] = decomposition.seasonal
data['residual'] = decomposition.resid

data[['trend', 'seasonal', 'residual']] = data[['trend', 'seasonal', 'residual']].ffill().bfill()

data_normalised = data.copy()
scaler = MinMaxScaler()
data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']] = scaler.fit_transform(
    data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']]
)
aux_data, test_data = train_test_split(data_normalised, test_size=0.1, shuffle = False)
train_data, validation_data = train_test_split(aux_data, test_size=0.2, shuffle = False)

train_tensor = torch.tensor(train_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)
validation_tensor = torch.tensor(validation_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)
test_tensor = torch.tensor(test_data[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']].values, dtype=torch.float32)

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

def cosine_noise_schedule(timesteps):
    return torch.cos(torch.linspace(0, timesteps, timesteps) * (0.5 * torch.pi)) ** 2

input_dim = 32
hidden_dim = 32
K = 10
epochs = 30
alpha = cosine_noise_schedule(K)

model = DFModel(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)#.to(device)
df_training(model, train_sequences, validation_sequences, alpha, K, epochs)

