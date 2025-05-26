from flask import Flask, request, jsonify
import torch
import pickle
import os
import numpy as np

from models.Lstm import LSTMRegressor
from models.Transformer import TransformerRegressor

from utils.utils import get_device

app = Flask(__name__)

LSTM_MODEL_PATH = "trained_models/lstm/model"
TRANSFORMER_MODEL_PATH = "trained_models/transformer/model"
SCALER_PATH = "trained_models/lstm/scaler"

INPUT_DIM = 64
HIDDEN_DIM = 512

device = get_device()

def load_model(model_type: str, house_id: int):
    model_path = {
        "lstm": os.path.join(LSTM_MODEL_PATH, f"H{house_id}.pth"),
        "transformer": os.path.join(TRANSFORMER_MODEL_PATH, f"H{house_id}.pth")
    }[model_type]

    model = {
        "lstm": LSTMRegressor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM),
        "transformer": TransformerRegressor(input_dim=INPUT_DIM, d_model=HIDDEN_DIM)
    }[model_type]

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_scaler(house_id: int):
    scaler_path = os.path.join(SCALER_PATH, f"H{house_id}.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    house_id = int(data.get("house_id"))
    model_type = data.get("model_type", "lstm")

    if not 1 <= house_id <= 20:
        return jsonify({"error": "Invalid house ID"}), 400

    try:
        model = load_model(model_type, house_id)
        scaler = load_scaler(house_id)

        # Dummy input: Replace with actual input sequence logic
        dummy_input = torch.zeros((1, 24, INPUT_DIM)).to(device).double()
        with torch.no_grad():
            forecast = model(dummy_input).cpu().numpy().tolist()

        return jsonify({
            "house": f"H{house_id}",
            "model": model_type,
            "forecast": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
