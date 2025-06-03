from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
import torch
import pickle
import os

from models.Lstm import LSTMRegressor
from models.Transformer import TransformerRegressor
from flask_application.forecast import forecast_day_from_model_aep, forecast_day_from_model_h
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

LSTM_MODEL_PATH = "../trained_models/lstm/model/"
TRANSFORMER_MODEL_PATH = "../trained_models/transformer/model/"
SCALER_PATH = "../trained_models/lstm/scaler/"

INPUT_DIM = 64
HIDDEN_DIM = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_scaler(house_id: int):
    scaler_path = os.path.join(SCALER_PATH, f"H{house_id}.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


@app.route("/forecast-day", methods=["POST"])
def forecast_day():
    data = request.json
    house_id = data.get("house_id")
    date_str = data.get("date")
    hidden_dim = 512
    if house_id == "AEP":
        input_dim = 5
    else:
        input_dim = 42
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400

    try:
        model_lstm_path = os.path.join(LSTM_MODEL_PATH, f"{house_id}.pth")
        model_transformer_path = os.path.join(TRANSFORMER_MODEL_PATH, f"{house_id}.pth")
        scaler_path = os.path.join(SCALER_PATH, f"{house_id}.pkl")
        csv_path = f"../training sets/{house_id}_Wh.csv"

        lstm_model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim)

        transformer_model = TransformerRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim)
        if house_id == "AEP":
            lstm_result_consumption = forecast_day_from_model_aep(
                lstm_model, target_date, csv_path, model_path=model_lstm_path, scaler_path=scaler_path, feature_index=0
            )

            transformer_result_consumption = forecast_day_from_model_aep(
                transformer_model, target_date, csv_path, model_path=model_transformer_path, scaler_path=scaler_path, feature_index=0
            )
        else:
            lstm_result_consumption = forecast_day_from_model_h(
                lstm_model, target_date, csv_path, model_path=model_lstm_path, scaler_path=scaler_path, feature_index=0
            )

            transformer_result_consumption = forecast_day_from_model_h(
                transformer_model, target_date, csv_path, model_path=model_transformer_path, scaler_path=scaler_path,
                feature_index=0
            )

            lstm_result_production = forecast_day_from_model_h(
                lstm_model, target_date, csv_path, model_path=model_lstm_path, scaler_path=scaler_path, feature_index=1
            )

            transformer_result_production = forecast_day_from_model_h(
                transformer_model, target_date, csv_path, model_path=model_transformer_path, scaler_path=scaler_path,
                feature_index=1
            )


        if house_id == "AEP":
            return jsonify({
                "date": date_str,
                "house": house_id.replace("_", " "),
                "true_values_consumption": lstm_result_consumption["true_values"],
                "true_values_production": None,
                "lstm_predictions_consumption": lstm_result_consumption["predictions"],
                "lstm_r2_consumption": lstm_result_consumption["r2_score"],
                "lstm_smape_consumption": lstm_result_consumption["smape"],
                "lstm_crps_consumption": lstm_result_consumption["crps"],

                "transformer_predictions_consumption": transformer_result_consumption["predictions"],
                "transformer_r2_consumption": transformer_result_consumption["r2_score"],
                "transformer_smape_consumption": transformer_result_consumption["smape"],
                "transformer_crps_consumption": transformer_result_consumption["crps"],

                "lstm_predictions_production": None,
                "lstm_r2_production": None,
                "lstm_smape_production": None,
                "lstm_crps_production": None,

                "transformer_predictions_production": None,
                "transformer_r2_production": None,
                "transformer_smape_production": None,
                "transformer_crps_production": None
            })
        else:
            return jsonify({
                "date": date_str,
                "house": house_id.replace("_", " "),
                "true_values_consumption": lstm_result_consumption["true_values"],
                "true_values_production": lstm_result_production["true_values"],

                "lstm_predictions_consumption": lstm_result_consumption["predictions"],
                "lstm_r2_consumption": lstm_result_consumption["r2_score"],
                "lstm_smape_consumption": lstm_result_consumption["smape"],
                "lstm_crps_consumption": lstm_result_consumption["crps"],

                "transformer_predictions_consumption": transformer_result_consumption["predictions"],
                "transformer_r2_consumption": transformer_result_consumption["r2_score"],
                "transformer_smape_consumption": transformer_result_consumption["smape"],
                "transformer_crps_consumption": transformer_result_consumption["crps"],

                "lstm_predictions_production": lstm_result_production["predictions"],
                "lstm_r2_production": lstm_result_production["r2_score"],
                "lstm_smape_production": lstm_result_production["smape"],
                "lstm_crps_production": lstm_result_production["crps"],

                "transformer_predictions_production": transformer_result_production["predictions"],
                "transformer_r2_production": transformer_result_production["r2_score"],
                "transformer_smape_production": transformer_result_production["smape"],
                "transformer_crps_production": transformer_result_production["crps"]
            })


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/stats/aep", methods=["GET"])
def get_aes_stats():
    data = request.json
    house_id = data.get("house_id")
    if not house_id:
        return jsonify({"error": "Missing house_id"}), 400

    try:
        csv_path = f"../training sets/{house_id}_Wh.csv"
        df = pd.read_csv(csv_path)
        consumption = df.iloc[:, 1]

        result = {
            "house": house_id,
            "consumption_mean": consumption.mean(),
            "consumption_baseline": consumption.iloc[0],
            "consumption_median": consumption.median()
        }
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/house_stats", methods=["POST"])
def get_h_stats():
    data = request.json
    house_id = data.get("house_id")
    if not house_id:
        return jsonify({"error": "Missing house_id"}), 400

    try:
        csv_path = f"../training sets/{house_id}_Wh.csv"
        df = pd.read_csv(csv_path)

        if house_id == "AEP":
            consumption = df.iloc[:, 1]
            result = {
                "house": house_id,
                "consumption_mean": consumption.mean(),
                "consumption_baseline": consumption.iloc[0],
                "consumption_median": consumption.median(),
                "production_mean": None,
                "production_baseline": None,
                "production_median": None,
                "consumption_series": consumption.tolist(),
                "production_series": None
            }
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            hourly_df = df.resample("1H").sum()

            consumption = hourly_df.iloc[:, 3]  # column 4 originally
            production = hourly_df.iloc[:, 2]   # column 3 originally

            result = {
                "house": house_id,
                "consumption_mean": consumption.mean(),
                "consumption_baseline": consumption.iloc[0],
                "consumption_median": consumption.median(),
                "production_mean": production.mean(),
                "production_baseline": production.iloc[0],
                "production_median": production.median(),
                "consumption_series": consumption.tolist(),
                "production_series": production.tolist()
            }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/house_day_plot", methods=["POST"])
def plot_house_day():
    import numpy as np

    data = request.json
    house_id = data.get("house_id")
    date_str = data.get("date")

    if not house_id or not date_str:
        return jsonify({"error": "Missing house_id or date (format: YYYY-MM-DD)"}), 400

    try:
        date = pd.to_datetime(date_str).date()  # <- Ensure it's a pure date
        csv_path = f"../training sets/{house_id}_Wh.csv"
        df = pd.read_csv(csv_path)

        # Ensure datetime index is set
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

        if house_id == "AEP":
            df["date_only"] = df.index.date
            day_data = df[df["date_only"] == date]


            if day_data.empty:
                return jsonify({"error": f"No data available for {date_str}"}), 404

            consumption = day_data.iloc[:, 0]

            result = {
                "house": house_id,
                "date": date_str,
                "consumption_mean": float(consumption.mean()),
                "consumption_baseline": float(consumption.iloc[0]),
                "consumption_median": float(consumption.median()),
                "production_mean": None,
                "production_baseline": None,
                "production_median": None,
                "consumption_series": np.round(consumption.values, 5).tolist(),
                "production_series": None
            }

        else:
            hourly_df = df.resample("1h").sum()
            hourly_df["date_only"] = hourly_df.index.date
            day_data = hourly_df[hourly_df["date_only"] == date]


            if day_data.empty:
                return jsonify({"error": f"No data available for {date_str}"}), 404

            consumption = day_data.iloc[:, 3]  # originally column 4
            production = day_data.iloc[:, 2]   # originally column 3

            result = {
                "house": house_id,
                "date": date_str,
                "consumption_mean": float(consumption.mean()),
                "consumption_baseline": float(consumption.iloc[0]),
                "consumption_median": float(consumption.median()),
                "production_mean": float(production.mean()),
                "production_baseline": float(production.iloc[0]),
                "production_median": float(production.median()),
                "consumption_series": np.round(consumption.values, 5).tolist(),
                "production_series": np.round(production.values, 5).tolist()
            }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)

