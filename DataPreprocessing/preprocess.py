import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from holidays.countries.ireland import Ireland
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(file_path, granularity="1h", add_noise=None):
    data = pd.read_csv(file_path)
    data = data[['date', ' Consumption(Wh)', ' Production(Wh)']]
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data = data.resample(granularity).sum()

    decomposition = seasonal_decompose(data[' Consumption(Wh)'], model='additive', period=24)
    data['trend'] = decomposition.trend.fillna(0).cumsum()
    data['seasonal'] = decomposition.seasonal.fillna(0).cumsum()
    data['residual'] = decomposition.resid.fillna(0).cumsum()

    decomposition_production = seasonal_decompose(data[' Production(Wh)'], model='additive', period=24)
    data['trend_production'] = decomposition_production.trend.fillna(0).cumsum()
    data['seasonal_production'] = decomposition_production.seasonal.fillna(0).cumsum()
    data['residual_production'] = decomposition_production.resid.fillna(0).cumsum()

    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['day_of_week'] = data.index.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

    ie_holidays = Ireland()
    data['is_holiday'] = data.index.to_series().apply(lambda x: int(x in ie_holidays))

    for lag in [1, 24, 48, 72, 168]:
        data[f'lag_{lag}'] = data[' Consumption(Wh)'].shift(lag)

    for window in [24, 48, 72, 168]:
        data[f'rolling_mean_{window}'] = data[' Consumption(Wh)'].rolling(window=window).mean()
        data[f'rolling_std_{window}'] = data[' Consumption(Wh)'].rolling(window=window).std()

    for lag in [1, 24, 48, 72, 168]:
        data[f'prod_lag_{lag}'] = data[' Production(Wh)'].shift(lag)

    for window in [24, 48, 72, 168]:
        data[f'prod_rolling_mean_{window}'] = data[' Production(Wh)'].rolling(window=window).mean()
        data[f'prod_rolling_std_{window}'] = data[' Production(Wh)'].rolling(window=window).std()

    day_of_year = data.index.dayofyear
    data['sin_day'] = np.sin(2 * np.pi * day_of_year / 365)
    data['cos_day'] = np.cos(2 * np.pi * day_of_year / 365)

    data = data.fillna(0)
    if add_noise:
        data = add_label_noise(data, label_col=' Consumption(Wh)', fraction_noisy=0.1)
    features_to_normalize = [
        ' Consumption(Wh)', ' Production(Wh)',
        'trend', 'seasonal', 'residual', 'trend_production', 'seasonal_production', 'residual_production',
        'year', 'month', 'day', 'day_of_week', 'is_weekend', 'is_holiday',
        'sin_day', 'cos_day'
    ]

    features_to_normalize += [col for col in data.columns if ('lag' in col or 'rolling' in col)]

    scaler = MinMaxScaler()
    data_normalized = pd.DataFrame(
        scaler.fit_transform(data[features_to_normalize]),
        columns=features_to_normalize,
        index=data.index
    )
    return data_normalized, scaler

def create_tensors(data, test_size=0.1, val_size=0.2):
    aux_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(aux_data, test_size=val_size, shuffle=False)
    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_data.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    test_start_index = data.index[-len(test_tensor)]
    test_end_index = data.index[-1]

    print(f"Test set starts at: {test_start_index}")
    print(f"Test set ends at: {test_end_index}")
    return train_tensor, val_tensor, test_tensor

def add_label_noise(df, label_col, fraction_noisy=0.1):
    np.random.seed(42)
    df_noisy = df.copy()
    num_rows = len(df_noisy)
    num_noisy = int(num_rows * fraction_noisy)
    noise_std = df_noisy[label_col].std() * 0.05
    noisy_indices = np.random.choice(df_noisy.index, size=num_noisy, replace=False)
    noise = np.random.normal(0, noise_std, num_noisy)

    df_noisy.loc[noisy_indices, label_col] += noise

    return df_noisy
def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length + 1):
        seq = data_tensor[i:i+seq_length]
        sequences.append(seq)
    return sequences

def plot_feature_correlation_heatmap(dataframe, save_dir="plots", title="Feature Correlation Heatmap"):
    os.makedirs(save_dir, exist_ok=True)
    corr_matrix = dataframe.corr()
    plt.figure(figsize=(22, 18))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    plt.title(title, fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    save_file = os.path.join(save_dir, "feature_correlation_heatmap.png")
    plt.savefig(save_file, dpi=300)
    # plt.show(block=False)

def preprocess_aep_dataset(file_path, test_size=0.1, val_size=0.2):
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = df.groupby('Datetime').mean()
    df = df.asfreq('1h')

    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month

    feature_cols = ['AEP_MW', 'hour', 'day', 'weekday', 'month']

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df = df.dropna()

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=val_size, shuffle=False)

    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_df.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)
    dataset_start_index = df.index[0]
    test_start_index = df.index[-len(test_tensor)]
    test_end_index = df.index[-1]

    print(f"dataset starts at: {dataset_start_index}")
    print(f"Test set starts at: {test_start_index}")
    print(f"Test set ends at: {test_end_index}")

    return train_tensor, val_tensor, test_tensor, scaler
