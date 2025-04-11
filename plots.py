import matplotlib
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

from utils import plotting_preprocess_xt, plotting_preprocess_epsilon

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self):
        self.mse_loss_list = []
        self.r2_score_list = []
        self.smape_score_list = []

        self.save_dir = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
        os.makedirs(self.save_dir, exist_ok=True)

    def update_metrics(self, mse, r2, smape):
        self.mse_loss_list.append(mse)
        self.r2_score_list.append(r2)
        self.smape_score_list.append(smape)

    def plot_and_save(self, x, y, label, ylabel, title, filename):
        plt.figure(figsize=(12, 4))
        plt.plot(x, y, label=label, linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, filename))  # Save plot
        plt.show()

    def plot_metrics(self):
        epochs = range(1, len(self.mse_loss_list) + 1)
        self.plot_and_save(epochs, self.mse_loss_list, "MSE Loss", "MSE Loss", "Training MSE Loss Over Epochs", "mse_loss.png")
        self.plot_and_save(epochs, self.r2_score_list, "R² Score", "R² Score", "Training R² Score Over Epochs", "r2_score.png")
        self.plot_and_save(epochs, self.smape_score_list, "SMAPE (%)", "SMAPE (%)", "Training SMAPE Over Epochs", "smape.png")

def plot_test_predictions(test_results, scaler):
    save_dir = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_dir, exist_ok=True)

    true_flat = []
    pred_flat = []
    epsilon_true_all, epsilon_pred_all = [], []

    for idx, (xt_true, xt_pred, epsilon_pred, epsilon_true) in enumerate(test_results):
        # xt_true = scaler.inverse_transform(xt_true)
        # xt_pred = scaler.inverse_transform(xt_pred)

        true_vals = xt_true[:, 3]
        pred_vals = xt_pred[:, 3]

        if idx < len(test_results) - 1:
            true_flat.append(true_vals[0])
            pred_flat.append(pred_vals[0])
        else:
            true_flat.extend(true_vals)
            pred_flat.extend(pred_vals)

        epsilon_true_all.append(epsilon_true)
        epsilon_pred_all.append(epsilon_pred)

    # Plot xt
    trace_true = go.Scatter(y=true_flat, mode='lines', name='True Consumption')
    trace_pred = go.Scatter(y=pred_flat, mode='lines', name='Predicted Consumption')
    fig_xt = go.Figure(data=[trace_true, trace_pred])
    fig_xt.update_layout(
        title="Flattened True vs. Predicted Consumption (Correctly Joined)",
        xaxis_title="Time Steps",
        yaxis_title="Consumption (Wh)",
        legend=dict(x=0, y=1.1, orientation='h'),
    )
    pio.write_html(fig_xt, file=os.path.join(save_dir, "xt_consumption_plotly.html"), auto_open=True)

    # Plot noise
    epsilon_true_flat, epsilon_pred_flat = plotting_preprocess_epsilon(epsilon_true_all, epsilon_pred_all)
    trace_eps_true = go.Scatter(y=epsilon_true_flat, mode='lines', name='True Noise (Avg)', line=dict(color='green'))
    trace_eps_pred = go.Scatter(y=epsilon_pred_flat, mode='lines', name='Predicted Noise (Avg)', line=dict(color='orange'))
    fig_eps = go.Figure(data=[trace_eps_true, trace_eps_pred])
    fig_eps.update_layout(
        title="Averaged Noise over All Features",
        xaxis_title="Time Steps",
        yaxis_title="Avg Noise",
        legend=dict(x=0, y=1.1, orientation='h'),
    )
    pio.write_html(fig_eps, file=os.path.join(save_dir, "epsilon_noise_plotly.html"), auto_open=True)
