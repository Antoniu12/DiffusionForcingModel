import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from df_training_v2 import DFModel, df_training
from sklearn.preprocessing import MinMaxScaler

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sa nu incerc pe H13!!!!

file_path = './H8_Wh.csv'
data = pd.read_csv(file_path)

data = data[['date', ' Consumption(Wh)']]

data['date'] = pd.to_datetime(data['date'])
data['numeric_date'] = (data['date'] - data['date'].min()).dt.days

data.set_index('date', inplace=True)
data = data.resample('5h').sum()


data_normalised = data.copy()
#min max scaler si la date
scaler = MinMaxScaler()
data_normalised[['numeric_date',' Consumption(Wh)']] = scaler.fit_transform(data_normalised[['numeric_date', ' Consumption(Wh)']])

train_data, test_data = train_test_split(data_normalised, test_size=0.1, shuffle = False)
#feature engineering si data analisys
#pearson corelation
train_tensor = torch.tensor(train_data[['numeric_date', ' Consumption(Wh)']].values, dtype=torch.float32)
test_tensor = torch.tensor(test_data[['numeric_date', ' Consumption(Wh)']].values, dtype=torch.float32)

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

input_dim = 32
hidden_dim = 16
K = 10
epochs = 10
alpha = cosine_noise_schedule(K)

model = DFModel(input_dim=input_dim, hidden_dim=hidden_dim)#.to(device)
df_training(model, train_sequences, alpha, K, epochs)
