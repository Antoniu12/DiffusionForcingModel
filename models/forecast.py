import torch
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta
from models.generate_predictions import predict_next_24h  # adjust as needed

def forecast_day_from_model(model, target_date, csv_path, model_path=None, scaler_path=None, feature_index=0, hidden_dim=512, seq_length=24):
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

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = predict_next_24h(model, context_seq, steps=25, device=device)
    pred_np = predictions.cpu().numpy()

    context_last = context_seq[-1].cpu().numpy()
    pred_aep = []
    for i in range(24):
        modified = context_last.copy()
        modified[feature_index] = pred_np[i, feature_index]
        pred_aep.append(scaler.inverse_transform([modified])[0][feature_index])

    print(true_values)
    plt.figure(figsize=(12, 5))
    plt.plot(pred_aep, label="Predicted AEP_MW", linestyle="--")
    plt.plot(true_values, label="True AEP_MW", linestyle="-")
    plt.title(f"AEP_MW Forecast vs Actual on {target_date.date()}")
    plt.xlabel("Hour")
    plt.ylabel("AEP_MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

