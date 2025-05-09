import os
from datetime import datetime

import holidays
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from holidays.countries.ireland import Ireland

def load_and_preprocess_data(file_path, granularity="D"):
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
    return train_tensor, val_tensor, test_tensor


def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        seq = data_tensor[i:i+seq_length]
        sequences.append(seq)
    return sequences

import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_feature_correlation_heatmap(dataframe, save_dir="plots", title="Feature Correlation Heatmap"):
    """
    Creates and saves a heatmap of the feature correlations with enlarged figure and font size.

    Args:
        dataframe (pd.DataFrame): Input dataframe after preprocessing (normalized or not).
        save_dir (str): Directory where to save the heatmap image.
        title (str): Title of the heatmap.

    Returns:
        None
    """
    # Create plots directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Compute correlation matrix
    corr_matrix = dataframe.corr()

    # Set up the matplotlib figure (BIG size)
    plt.figure(figsize=(22, 18))  # ✅ Much bigger figure

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,           # ✅ Write the numbers
        fmt=".2f",             # ✅ 2 decimals
        linewidths=0.5,
        annot_kws={"size": 9}  # ✅ Bigger font size for numbers
    )

    plt.title(title, fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)   # ✅ Rotate and align x-labels
    plt.yticks(rotation=0, fontsize=10)                # ✅ Y labels stay horizontal
    plt.tight_layout()

    # Save and show
    save_file = os.path.join(save_path, "feature_correlation_heatmap.png")
    plt.savefig(save_file, dpi=300)
    plt.show(block=False)


