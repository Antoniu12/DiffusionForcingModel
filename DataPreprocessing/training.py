import os
import pickle
from datetime import datetime, timedelta

import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from DiffusionBase.DF_Backbone_NextToken import DFBackbone_NextToken
from DiffusionBase.DfTraining import train_next_token_diffusion, rolling_next_token_prediction, \
    predict_with_random_last_noise, predict_with_random_last_noise_2, autoregressive_forecast
from flask_application.GeneratePredictions import autoregressive_next_24
from models.Lstm import LSTMRegressor
from models.TrainModel import train_model, evaluate_test_dataset
from models.Transformer import TransformerRegressor
from utils.Sequence_Dataset import SequenceDataset

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from utils.utils import save_model, flatten_overlapping_windows_preds, flatten_overlapping_windows_targets
import torch


from DataPreprocessing.preprocess import load_and_preprocess_data, create_sequences, create_tensors, \
    preprocess_aep_dataset
from utils import utils
from utils.plots import plot_test_predictions

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
os.makedirs(save_path, exist_ok=True)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    diff = np.abs(y_pred - y_true)
    return 100 * np.mean(2.0 * diff / denominator)

def mape(y_true, y_pred):
    y_true_safe = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true_safe))

def train_diffusion_on_h_dataset(dataset_number):
    file_path = f'./training sets/H{dataset_number}_Wh.csv'
    save_path = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_path, exist_ok=True)

    data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
    train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)

    input_dim = data_normalised.shape[1]
    hidden_dim = input_dim
    K = 1000
    epochs = 100
    betas = utils.cosine_beta_schedule(K)
    alpha, alpha_bar = utils.get_alphas(betas)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    seq_length = 24

    train_sequences = create_sequences(train_tensor, seq_length)
    test_sequences = create_sequences(test_tensor, seq_length)
    validation_sequences = create_sequences(validation_tensor, seq_length)
    print(f"Number of training sequences: {len(train_sequences)}")
    print(f"Number of validation sequences: {len(validation_sequences)}")
    print(f"Number of test sequences: {len(test_sequences)}")


    model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
    model = model.to(device)
    train_next_token_diffusion(model, train_sequences, validation_sequences, alpha, alpha_bar, K, epochs, device=device)
    save_model(model, save_path, )

    test_results = rolling_next_token_prediction(model, test_tensor, alpha, alpha_bar, K, device=device)
    plot_test_predictions(test_results, scaler, save_path)
    predictions = predict_with_random_last_noise(
        model=model,
        test_tensor=test_sequences,
        alpha=alpha,
        alpha_bar=alpha_bar,
        K=K,
        device=device
    )

    forecast = flatten_overlapping_windows_preds(predictions)
    true = flatten_overlapping_windows_targets(test_sequences)

    forecast_prod = forecast[:, 0]
    true_prod = true[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Consumption")
    plt.plot(forecast_prod, label="Forecasted Consumption")
    plt.title("One-Step Forecast vs True Consumption")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    forecast_prod = forecast[:, 1]
    true_prod = true[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Production")
    plt.plot(forecast_prod, label="Forecasted Production")
    plt.title("One-Step Forecast vs True Production")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Production")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_diffusion_on_aep_dataset():
    train_tensor, val_tensor, test_tensor, scaler = preprocess_aep_dataset("./training sets/AEP_Wh.csv")

    input_dim = train_tensor.shape[1]
    hidden_dim = input_dim
    K = 1000
    epochs = 100
    betas = utils.cosine_beta_schedule(K)
    alpha, alpha_bar = utils.get_alphas(betas)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    seq_length = 24

    train_sequences = create_sequences(train_tensor, seq_length)
    test_sequences = create_sequences(test_tensor, seq_length)
    validation_sequences = create_sequences(val_tensor, seq_length)
    print(f"Number of training sequences: {len(train_sequences)}")
    print(f"Number of validation sequences: {len(validation_sequences)}")
    print(f"Number of test sequences: {len(test_sequences)}")
    model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=seq_length)
    model = model.to(device)
    train_next_token_diffusion(model, train_sequences, validation_sequences, alpha, alpha_bar, K, epochs, device=device)
    save_model(model, save_path, )

    test_results = rolling_next_token_prediction(model, test_tensor, alpha, alpha_bar, K, device=device)
    plot_test_predictions(test_results, scaler, save_path)
    predictions = predict_with_random_last_noise(
        model=model,
        test_tensor=test_sequences,
        alpha=alpha,
        alpha_bar=alpha_bar,
        K=K,
        device=device
    )

    forecast = flatten_overlapping_windows_preds(predictions)
    true = flatten_overlapping_windows_targets(test_sequences)

    forecast_prod = forecast[:, 0]
    true_prod = true[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Consumption")
    plt.plot(forecast_prod, label="Forecasted Consumption")
    plt.title("One-Step Forecast vs True Consumption")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def tain_lstm_and_transformer():
    file_path = './training sets'

    trained_models_root = os.path.join(".", "trained_models")

    lstm_model_dir = os.path.join(trained_models_root, "lstm", "model")
    lstm_scaler_dir = os.path.join(trained_models_root, "lstm", "scaler")
    transformer_model_dir = os.path.join(trained_models_root, "transformer", "model")
    transformer_scaler_dir = os.path.join(trained_models_root, "transformer", "scaler")

    os.makedirs(lstm_model_dir, exist_ok=True)
    os.makedirs(lstm_scaler_dir, exist_ok=True)
    os.makedirs(transformer_model_dir, exist_ok=True)
    os.makedirs(transformer_scaler_dir, exist_ok=True)
    hidden_dim = 512
    epochs = 120
    seq_length = 24
    for i in range(1, 21):
        file_name = f"H{i}_Wh.csv"
        data_location = os.path.join(file_path, file_name)
        print(f"Processing: {data_location}")
        data_normalised, scaler = load_and_preprocess_data(data_location, "1h")
        train_tensor, validation_tensor, test_tensor = create_tensors(data_normalised, 0.1, 0.2)
        test_start_date = data_normalised.iloc[-len(test_tensor):].index[0]
        test_end_date = data_normalised.iloc[-1].name

        print(f"Test set covers from {test_start_date} to {test_end_date}")
        input_dim = data_normalised.shape[1]
        hidden_dim = 512
        epochs = 120
        seq_length = 24
        train_loader = DataLoader(SequenceDataset(train_tensor, seq_length), batch_size=32, shuffle=True)
        val_loader = DataLoader(SequenceDataset(validation_tensor, seq_length), batch_size=32, shuffle=False)
        test_loader = DataLoader(SequenceDataset(test_tensor, seq_length), batch_size=32, shuffle=False)

        lstm_model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)

        transformer_model = TransformerRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)

        train_model(lstm_model, train_loader, val_loader, num_epochs=epochs, device=device)

        torch.save(lstm_model.state_dict(), os.path.join(lstm_model_dir, f"H{i}.pth"))
        with open(os.path.join(lstm_scaler_dir, f"H{i}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        train_model(transformer_model, train_loader, val_loader, num_epochs=epochs, device=device)

        torch.save(transformer_model.state_dict(), os.path.join(transformer_model_dir, f"H{i}.pth"))
        with open(os.path.join(transformer_scaler_dir, f"H{i}.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    train_tensor, val_tensor, test_tensor, scaler = preprocess_aep_dataset("./training sets/AEP_Wh.csv")
    input_dim = train_tensor.shape[1]

    train_loader = DataLoader(SequenceDataset(train_tensor, seq_length), batch_size=32, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_tensor, seq_length), batch_size=32, shuffle=False)
    test_loader = DataLoader(SequenceDataset(test_tensor, seq_length), batch_size=32, shuffle=False)

    lstm_model = LSTMRegressor(input_dim, hidden_dim, input_dim)
    lstm_model.to(device)
    train_model(lstm_model, train_loader, val_loader, num_epochs=epochs, device=device)
    torch.save(lstm_model.state_dict(), os.path.join(lstm_model_dir, f"AEP.pth"))
    with open(os.path.join(lstm_scaler_dir, f"AEP.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    transformer_model = TransformerRegressor(input_dim, hidden_dim, input_dim)
    transformer_model.to(device)
    train_model(transformer_model, train_loader, val_loader, num_epochs=epochs, device=device)
    torch.save(transformer_model.state_dict(), os.path.join(transformer_model_dir, f"AEP.pth"))
    with open(os.path.join(transformer_scaler_dir, f"AEP.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def show_plot_for_diffusion():
    file_path = './training sets/H1_Wh.csv'

    data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
    _, _, test_tensor = create_tensors(data_normalised, 0.1, 0.2)
    test_sequences = create_sequences(test_tensor, 24)

    input_dim = data_normalised.shape[1]
    hidden_dim = input_dim
    K = 1000
    betas = utils.cosine_beta_schedule(K)
    alpha, alpha_bar = utils.get_alphas(betas)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=24)
    model = model.to(device)
    model.load_state_dict(torch.load("./trained_models/diffusion/model/H1_noisy.pth", map_location=device))

    st = time.time()
    results = predict_with_random_last_noise_2(
        model=model,
        test_tensor=test_sequences,
        alpha=alpha,
        alpha_bar=alpha_bar,
        K=K,
        device=device,
        scaler=scaler,
        feature_index=0
    )
    end = time.time()
    print("time needed: ", end - st)
    true_prod = results['targets']
    forecast_prod = results['preds']

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Consumption")
    plt.plot(forecast_prod, label="Forecasted Consumption")
    plt.title("Diffusion Forecast vs True Consumption")
    plt.xlabel("Time step")
    plt.ylabel("Energy (W/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Consumption Metrics:")
    print("MAE:", results['MAE'])
    print("MSE:", results['MSE'])
    print("R2:", results['R2'])
    print("SMAPE:", results['SMAPE'])
    print("MAPE:", results['MAPE'])

    st = time.time()
    results_prod = predict_with_random_last_noise_2(
        model=model,
        test_tensor=test_sequences,
        alpha=alpha,
        alpha_bar=alpha_bar,
        K=K,
        device=device,
        scaler=scaler,
        feature_index=1
    )
    end = time.time()
    print("time needed: ", end - st)
    true_prod = results_prod['targets']
    forecast_prod = results_prod['preds']

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Production")
    plt.plot(forecast_prod, label="Forecasted Production")
    plt.title("Diffusion Forecast vs True Production")
    plt.xlabel("Time step")
    plt.ylabel("Energy (W/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Production Metrics:")
    print("MAE:", results_prod['MAE'])
    print("MSE:", results_prod['MSE'])
    print("R2:", results_prod['R2'])
    print("SMAPE:", results_prod['SMAPE'])
    print("MAPE:", results_prod['MAPE'])

def show_plot_for_LT(m):
    file_path = './training sets/H1_Wh.csv'
    data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
    _, _, test_tensor = create_tensors(data_normalised, 0.1, 0.2)

    input_dim = data_normalised.shape[1]
    hidden_dim = 512

    test_loader = DataLoader(SequenceDataset(test_tensor, 24), batch_size=32, shuffle=False)

    if m == "LSTM":
        model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
        model = model.to(device)
        model.load_state_dict(torch.load("./trained_models/lstm/model/H1_noisy.pth", map_location=device))
    else:
        model = TransformerRegressor(input_dim, hidden_dim, input_dim)
        model = model.to(device)
        model.load_state_dict(torch.load("./trained_models/transformer/model/H1_noisy.pth", map_location=device))

    st = time.time()
    results = evaluate_test_dataset(model, test_loader, scaler, feature_index=0, device=device)
    end = time.time()
    print("time needed: ", end - st)
    true_prod = results['targets']
    forecast_prod = results['preds']

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Consumption")
    plt.plot(forecast_prod, label="Forecasted Consumption")
    plt.title(f"{m} Forecast vs True Consumption")
    plt.xlabel("Time step")
    plt.ylabel("Energy (W/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Consumption Metrics:")
    print("MAE:", results['MAE'])
    print("MSE:", results['MSE'])
    print("R2:", results['R2'])
    print("SMAPE:", results['SMAPE'])
    print("MAPE:", results['MAPE'])

    st = time.time()
    results_prod = evaluate_test_dataset(model, test_loader, scaler, feature_index=1, device=device)
    end = time.time()
    print("time needed: ", end - st)
    true_prod = results_prod['targets']
    forecast_prod = results_prod['preds']

    plt.figure(figsize=(10, 5))
    plt.plot(true_prod, label="True Production")
    plt.plot(forecast_prod, label="Forecasted Production")
    plt.title(f"{m} Forecast vs True Production")
    plt.xlabel("Time step")
    plt.ylabel("Energy (W/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Production Metrics:")
    print("MAE:", results_prod['MAE'])
    print("MSE:", results_prod['MSE'])
    print("R2:", results_prod['R2'])
    print("SMAPE:", results_prod['SMAPE'])
    print("MAPE:", results_prod['MAPE'])

def autoregressive_LT(m):
        file_path = './training sets/H2_Wh.csv'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_date = pd.Timestamp("2020-12-29 00:00:00")
        start_time = target_date - timedelta(days=1)
        end_time = target_date + timedelta(hours=23)
        raw_df = pd.read_csv(file_path, parse_dates=["date"])
        raw_df.set_index("date", inplace=True)
        raw_df = raw_df.sort_index()
        raw_df = raw_df.resample("1h").sum()

        true_window = raw_df.loc[start_time:end_time]
        label = " Consumption(Wh)"


        true_consumption = true_window[label].iloc[24:].values

        data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
        start_time = pd.Timestamp(target_date) - timedelta(hours=24)
        end_time = pd.Timestamp(target_date) + timedelta(hours=23)

        window_df = data_normalised.loc[start_time:end_time].copy()
        assert len(window_df) == 48, f"Expected 48 rows, got {len(window_df)}"

        window_tensor = torch.tensor(window_df.values, dtype=torch.float32).to(device)

        context_tensor = window_tensor[:24]
        ground_truth_seq = window_tensor[24:48]

        input_dim = data_normalised.shape[1]
        hidden_dim = 512
        if m == "LSTM":
            model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
            model = model.to(device)
            model.load_state_dict(torch.load("./trained_models/lstm/model/H2.pth", map_location=device))
        else:
            model = TransformerRegressor(input_dim, hidden_dim, input_dim)
            model = model.to(device)
            model.load_state_dict(torch.load("./trained_models/transformer/model/H2.pth", map_location=device))

        model.to(device)
        model.eval()
        st = time.time()
        predictions = autoregressive_next_24(model, context_tensor, steps=24, device=device)
        end = time.time()
        print("time needed: ", end - st)
        pred_np = predictions.cpu().numpy()

        context_last = context_tensor[-1].cpu().numpy()
        pred_consumption = []
        for i in range(24):
            modified = context_last.copy()
            clamped_scaled = np.clip(pred_np[i, 0], 0.0, 1.0)
            modified[0] = clamped_scaled
            denorm_value = scaler.inverse_transform([modified])[0][0]
            pred_consumption.append(denorm_value)
        pred_consumption = np.array(pred_consumption)

        mae_val = mean_absolute_error(true_consumption, pred_consumption)
        mse_val = mean_squared_error(true_consumption, pred_consumption)
        r2 = r2_score(true_consumption, pred_consumption)
        smape_val = smape(true_consumption, pred_consumption)
        mape_val = mape(true_consumption, pred_consumption)

        plt.figure(figsize=(10, 5))
        plt.plot(true_consumption, label="True")
        plt.plot(pred_consumption, label="Forecasted")
        plt.title(f"{m} Forecast vs True Energy")
        plt.xlabel("Time step")
        plt.ylabel("Energy (W/h)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Consumption Metrics:")
        print("MAE:", mae_val)
        print("MSE:", mse_val)
        print("R2:", r2)
        print("SMAPE:", smape_val)
        print("MAPE:", mape_val)

def autoregressive_diffusion():
    file_path = './training sets/H2_Wh.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_date = pd.Timestamp("2020-12-29 00:00:00")
    start_time = target_date - timedelta(days=1)
    end_time = target_date + timedelta(hours=23)
    raw_df = pd.read_csv(file_path, parse_dates=["date"])
    raw_df.set_index("date", inplace=True)
    raw_df = raw_df.sort_index()
    raw_df = raw_df.resample("1h").sum()

    true_window = raw_df.loc[start_time:end_time]
    label = " Consumption(Wh)"

    true_consumption = true_window[label].iloc[24:].values

    data_normalised, scaler = load_and_preprocess_data(file_path, "1h")
    start_time = pd.Timestamp(target_date) - timedelta(hours=24)
    end_time = pd.Timestamp(target_date) + timedelta(hours=23)

    window_df = data_normalised.loc[start_time:end_time].copy()
    assert len(window_df) == 48, f"Expected 48 rows, got {len(window_df)}"

    window_tensor = torch.tensor(window_df.values, dtype=torch.float32).to(device)

    context_tensor = window_tensor[:24]
    ground_truth_seq = window_tensor[24:48]

    input_dim = data_normalised.shape[1]
    hidden_dim = input_dim
    K = 1000
    betas = utils.cosine_beta_schedule(K)
    alpha, alpha_bar = utils.get_alphas(betas)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    model = DFBackbone_NextToken(input_dim=input_dim, hidden_dim=hidden_dim, seq_dim=24)

    model.load_state_dict(torch.load("./trained_models/diffusion/model/H2.pth", map_location=device))

    model.to(device)
    model.eval()
    for i in range(10):
        st = time.time()
        predictions = autoregressive_forecast(
            model, context_tensor, alpha, alpha_bar, K, device=device, steps=24
        )
        end = time.time()
        print("time needed: ", end - st)
        pred_np = predictions.cpu().numpy()

        context_last = context_tensor[-1].cpu().numpy()
        pred_consumption = []
        for i in range(24):
            modified = context_last.copy()
            clamped_scaled = np.clip(pred_np[i, 0], 0.0, 1.0)
            modified[0] = clamped_scaled
            denorm_value = scaler.inverse_transform([modified])[0][0]
            pred_consumption.append(denorm_value)
        pred_consumption = np.array(pred_consumption)

        mae_val = mean_absolute_error(true_consumption, pred_consumption)
        mse_val = mean_squared_error(true_consumption, pred_consumption)
        r2 = r2_score(true_consumption, pred_consumption)
        smape_val = smape(true_consumption, pred_consumption)
        mape_val = mape(true_consumption, pred_consumption)
        print("Consumption Metrics:")
        print("MAE:", mae_val)
        print("MSE:", mse_val)
        print("R2:", r2)
        print("SMAPE:", smape_val)
        print("MAPE:", mape_val)
        plt.figure(figsize=(10, 5))
        plt.plot(true_consumption, label="True")
        plt.plot(pred_consumption, label="Forecasted")
        plt.title("Diffusion Forecast vs True Energy")
        plt.xlabel("Time step")
        plt.ylabel("Energy (W/h)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()