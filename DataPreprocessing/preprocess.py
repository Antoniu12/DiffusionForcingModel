import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def load_and_preprocess_data(file_path, granularity="D"):
    data = pd.read_csv(file_path)
    data = data[['date', ' Consumption(Wh)']]
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data.set_index('date', inplace=True)
    data = data.resample(granularity).sum()
    decomposition = seasonal_decompose(data[' Consumption(Wh)'], model='additive', period=24)

    data['trend'] = decomposition.trend.fillna(0).cumsum()
    data['seasonal'] = decomposition.seasonal.fillna(0).cumsum()
    data['residual'] = decomposition.resid.fillna(0).cumsum()

    data_normalised = data.copy()
    scaler = MinMaxScaler()
    data_normalised[
        ['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']] = scaler.fit_transform(
        data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']]
    )
    data_normalised = data_normalised[['year', 'month', 'day', ' Consumption(Wh)', 'trend', 'seasonal', 'residual']]
    return data_normalised, scaler

def create_tensors(data, test_size=0.1, val_size=0.2):

    aux_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(aux_data, test_size=val_size, shuffle=False)

    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_data.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    return train_tensor, val_tensor, test_tensor


def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        seq = data_tensor[i:i+seq_length]
        sequences.append(seq)
    return sequences