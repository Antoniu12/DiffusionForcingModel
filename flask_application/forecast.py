import numpy as np
import torch
import pandas as pd
import pickle
from datetime import timedelta

from properscoring import crps_ensemble
from sklearn.metrics import r2_score

from flask_application.generate_predictions import autoregressive_next_24, teacher_forcing_next_24
from DataPreprocessing.preprocess import load_and_preprocess_data
from DiffusionBase.df_training_next_token import autoregressive_forecast, teacher_forcing_forecast


def smape(y_true, y_pred):
    return 100 * np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)
    )

def forecast_day_from_model_aep(model, target_date, csv_path, model_path=None, scaler_path=None,
                                feature_index=0, hidden_dim=512, seq_length=24, mode="autoregressive"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = target_date - timedelta(days=1)
    end_time = target_date + timedelta(hours=23)
    all_data_df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    all_data_df = all_data_df.sort_values("Datetime").drop_duplicates("Datetime")
    mask = (all_data_df["Datetime"] >= start_time) & (all_data_df["Datetime"] <= end_time)
    subset_df = all_data_df.loc[mask].copy().reset_index(drop=True)
    assert len(subset_df) == 48, "Expected exactly 48 hourly data points (1 day input + 1 day target)."

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    subset_df["hour"] = subset_df["Datetime"].dt.hour
    subset_df["day"] = subset_df["Datetime"].dt.day
    subset_df["weekday"] = subset_df["Datetime"].dt.weekday
    subset_df["month"] = subset_df["Datetime"].dt.month

    feature_cols = ['AEP_MW', 'hour', 'day', 'weekday', 'month']
    scaled_input = scaler.transform(subset_df[feature_cols])
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

    context_seq = input_tensor[:seq_length]
    true_values = subset_df.iloc[seq_length:]["AEP_MW"].values
    ground_truth_seq = input_tensor[24:48]

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    if mode == "autoregressive":
        predictions = autoregressive_next_24(model, context_seq, steps=24, device=device)
    elif mode == "teacher_forcing":
        predictions = teacher_forcing_next_24(model, context_seq, ground_truth_seq, steps=24, device=device)
    pred_np = predictions.cpu().numpy()

    context_last = context_seq[-1].cpu().numpy()
    pred_aep = []
    for i in range(24):
        modified = context_last.copy()
        modified[feature_index] = pred_np[i, feature_index]
        denorm_value = scaler.inverse_transform([modified])[0][feature_index]
        clamped = max(0.0, denorm_value)
        pred_aep.append(clamped)

    pred_aep = np.array(pred_aep)

    r2 = r2_score(true_values, pred_aep)
    smape_val = smape(true_values, pred_aep)
    crps_val = crps_ensemble(true_values, np.expand_dims(pred_aep, axis=1)).mean()

    return {
        "predictions": pred_aep.tolist(),
        "true_values": true_values.tolist(),
        "r2_score": r2,
        "smape": smape_val,
        "crps": crps_val
    }

def forecast_day_from_model_h(model, target_date, csv_path, model_path=None, scaler_path=None,
                              feature_index=1, hidden_dim=512, seq_length=24, mode="autoregressive"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = target_date - timedelta(days=1)
    end_time = target_date + timedelta(hours=23)
    raw_df = pd.read_csv(csv_path, parse_dates=["date"])
    raw_df.set_index("date", inplace=True)
    raw_df = raw_df.sort_index()
    raw_df = raw_df.resample("1h").sum()

    true_window = raw_df.loc[start_time:end_time]
    if feature_index == 0:
        label = " Consumption(Wh)"
    else:
        label = " Production(Wh)"

    true_consumption = true_window[label].iloc[24:].values

    data_normalised, scaler = load_and_preprocess_data(csv_path, "1h")

    start_time = pd.Timestamp(target_date) - timedelta(hours=24)
    end_time = pd.Timestamp(target_date) + timedelta(hours=23)

    window_df = data_normalised.loc[start_time:end_time].copy()
    assert len(window_df) == 48, f"Expected 48 rows, got {len(window_df)}"

    window_tensor = torch.tensor(window_df.values, dtype=torch.float32).to(device)

    context_tensor = window_tensor[:24]
    ground_truth_seq = window_tensor[24:48]
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if mode == "autoregressive":
        predictions = autoregressive_next_24(model, context_tensor, steps=24, device=device)
    elif mode == "teacher_forcing":
        predictions = teacher_forcing_next_24(model, context_tensor, ground_truth_seq, steps=24, device=device)
    pred_np = predictions.cpu().numpy()

    context_last = context_tensor[-1].cpu().numpy()
    pred_consumption = []
    for i in range(24):
        modified = context_last.copy()
        clamped_scaled = np.clip(pred_np[i, feature_index], 0.0, 1.0)
        modified[feature_index] = clamped_scaled
        denorm_value = scaler.inverse_transform([modified])[0][feature_index]
        # clamped = max(0.0, denorm_value)
        pred_consumption.append(denorm_value)
    pred_consumption = np.array(pred_consumption)

    r2 = r2_score(true_consumption, pred_consumption)
    smape_val = smape(true_consumption, pred_consumption)
    crps_val = crps_ensemble(true_consumption, np.expand_dims(pred_consumption, axis=1)).mean()

    return {
        "predictions": pred_consumption.tolist(),
        "true_values": true_consumption.tolist(),
        "r2_score": r2,
        "smape": smape_val,
        "crps": crps_val
    }

def forecast_day_from_diffusion_h(
    model, target_date, csv_path, alpha, alpha_bar, K, model_path=None, scaler_path=None,
    feature_index=1, hidden_dim=512, seq_length=24, device="cpu", mode="autoregressive"
):
    start_time = target_date - timedelta(days=1)
    end_time = target_date + timedelta(hours=23)
    raw_df = pd.read_csv(csv_path, parse_dates=["date"])
    raw_df.set_index("date", inplace=True)
    raw_df = raw_df.sort_index()
    raw_df = raw_df.resample("1h").sum()

    true_window = raw_df.loc[start_time:end_time]
    label = " Consumption(Wh)" if feature_index == 0 else " Production(Wh)"
    true_values = true_window[label].iloc[24:].values

    data_normalised, scaler = load_and_preprocess_data(csv_path, "1h")

    start_time = pd.Timestamp(target_date) - timedelta(hours=24)
    end_time = pd.Timestamp(target_date) + timedelta(hours=23)
    window_df = data_normalised.loc[start_time:end_time].copy()
    assert len(window_df) == 48, f"Expected 48 rows, got {len(window_df)}"

    window_tensor = torch.tensor(window_df.values, dtype=torch.float32).to(device)
    context_tensor = window_tensor[:24]
    ground_truth_seq = window_tensor[24:48]

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    if mode == "autoregressive":
        predictions = autoregressive_forecast(
            model, context_tensor, alpha, alpha_bar, K, device=device, steps=24
        )
    elif mode == "teacher_forcing":
        predictions = teacher_forcing_forecast(
            model, context_tensor, ground_truth_seq, alpha, alpha_bar, K, device=device, steps=24
        )
    pred_np = predictions.cpu().numpy()

    context_last = context_tensor[-1].cpu().numpy()
    pred_list = []
    for i in range(24):
        modified = context_last.copy()
        clamped_scaled = np.clip(pred_np[i, feature_index], 0.0, 1.0)
        modified[feature_index] = clamped_scaled
        denorm_value = scaler.inverse_transform([modified])[0][feature_index]
        # clamped = max(0.0, denorm_value)
        pred_list.append(denorm_value)
    pred_list = np.array(pred_list)

    r2 = r2_score(true_values, pred_list)
    smape_val = smape(true_values, pred_list)
    crps_val = crps_ensemble(true_values, np.expand_dims(pred_list, axis=1)).mean()

    return {
        "predictions": pred_list.tolist(),
        "true_values": true_values.tolist(),
        "r2_score": r2,
        "smape": smape_val,
        "crps": crps_val
    }