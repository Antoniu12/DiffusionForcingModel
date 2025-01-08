import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from df_training_v2 import DFModel, df_training
from torch.nn.functional import cosine_similarity

#sa nu incerc pe H13!!!!

file_path = './H8_Wh.csv'
data = pd.read_csv(file_path)

data = data[['date', ' Consumption(Wh)']]
data['date'] = pd.to_datetime(data['date'])

data.set_index('date', inplace=True)
data = data.resample('D').sum()

data_normalised = data.copy()
data_normalised[' Consumption(Wh)'] = (data[' Consumption(Wh)'] - data[' Consumption(Wh)'].min()) / (data[' Consumption(Wh)'].max() - data[' Consumption(Wh)'].min())
#min max scaler si la date
train_data, test_data = train_test_split(data_normalised, test_size=0.1, shuffle = False)
#feature engineering si data analisys
#pearson corelation
train_tensor = torch.tensor(train_data[' Consumption(Wh)'].values, dtype=torch.float32)
test_tensor = torch.tensor(test_data[' Consumption(Wh)'].values, dtype=torch.float32)

def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        seq = data_tensor[i:i+seq_length]
        label = data_tensor[i+seq_length]
        sequences.append((seq, label))
    return sequences

seq_length = 24
train_sequences = create_sequences(train_tensor, seq_length)
test_sequences = create_sequences(test_tensor, seq_length)

print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}")

def cosine_noise_schedule(timesteps):
    return torch.cos(torch.linspace(0, timesteps, timesteps) * (0.5 * torch.pi)) ** 2

input_dim = 16
hidden_dim = 16
K = 10
epochs = 10
alpha = cosine_noise_schedule(K)

model = DFModel(input_dim=input_dim, hidden_dim=hidden_dim)
df_training(model, train_sequences, alpha, K, epochs)
