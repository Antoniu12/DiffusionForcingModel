from datetime import datetime

from flask import Flask, request, jsonify
import torch
import pickle
import os
import numpy as np

from models.Lstm import LSTMRegressor
from models.Transformer import TransformerRegressor
from models.forecast import forecast_day_from_model_aep, forecast_day_from_model_h
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

if __name__ == "__main__":
    app.run(debug=True)

