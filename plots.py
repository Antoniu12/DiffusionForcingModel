import matplotlib
import numpy as np
from datetime import datetime
import os
import plotly.graph_objs as go
import plotly.io as pio
import torch

from utils.utils import plotting_preprocess_epsilon

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self):
        self.mse_loss_list = []
        self.r2_score_list = []
        self.r2_score_listxt = []

        self.smape_score_list = []

        self.save_dir = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
        os.makedirs(self.save_dir, exist_ok=True)

    def update_metrics(self, mse, r2, r2xt, smape):
        self.mse_loss_list.append(mse.detach().cpu().item() if torch.is_tensor(mse) else mse)
        self.r2_score_list.append(r2.detach().cpu().item() if torch.is_tensor(r2) else r2)
        self.r2_score_listxt.append(r2xt.detach().cpu().item() if torch.is_tensor(r2xt) else r2xt)
        self.smape_score_list.append(smape.detach().cpu().item() if torch.is_tensor(smape) else smape)

    def plot_and_save(self, x, y, label, ylabel, title, filename):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        y = y.detach().cpu().numpy() if torch.is_tensor(y) else y
        plt.figure(figsize=(12, 4))
        plt.plot(x, y, label=label, linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.show(block=False)

    def plot_metrics(self):
        epochs = range(1, len(self.mse_loss_list) + 1)
        self.plot_and_save(epochs, self.mse_loss_list, "MSE Loss", "MSE Loss", "Training MSE Loss Over Epochs", "mse_loss.png")
        self.plot_and_save(epochs, self.r2_score_listxt, "R² Score xt ", "R² Score", "Training R² xt Score Over Epochs", "r2_score_xt.png")
        self.plot_and_save(epochs, self.r2_score_list, "R² Score epsilon", "R² Score", "Training R² epsilin Score Over Epochs", "r2_score.png")
        self.plot_and_save(epochs, self.smape_score_list, "SMAPE (%)", "SMAPE (%)", "Training SMAPE Over Epochs", "smape.png")

def plot_test_predictions(test_results, scaler):
    from collections import defaultdict
    import os
    import numpy as np
    import plotly.graph_objs as go
    import plotly.io as pio
    from datetime import datetime
    from utils.utils import plotting_preprocess_epsilon

    save_dir = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_dir, exist_ok=True)

    pred_dict_consumption = defaultdict(list)
    true_dict_consumption = {}

    pred_dict_production = defaultdict(list)
    true_dict_production = {}

    global_timestep = 0

    for idx, (xt_true, xt_pred, epsilon_pred, epsilon_true) in enumerate(test_results):
        seq_len = xt_true.shape[0]

        for i in range(seq_len):
            t = global_timestep + i
            pred_dict_consumption[t].append(xt_pred[i, 0])   # Corrected: 0 is Consumption
            pred_dict_production[t].append(xt_pred[i, 1])    # Corrected: 1 is Production

            if t not in true_dict_consumption:
                true_dict_consumption[t] = xt_true[i, 0]
            if t not in true_dict_production:
                true_dict_production[t] = xt_true[i, 1]

        global_timestep += 1

    timesteps = sorted(true_dict_consumption.keys())

    # Prepare Consumption Data
    true_flat = [true_dict_consumption[t] for t in timesteps]
    pred_first = [pred_dict_consumption[t][0] for t in timesteps]
    pred_min = [np.min(pred_dict_consumption[t]) for t in timesteps]
    pred_max = [np.max(pred_dict_consumption[t]) for t in timesteps]

    # Prepare Production Data
    true_flat_production = [true_dict_production[t] for t in timesteps]
    pred_first_production = [pred_dict_production[t][0] for t in timesteps]
    pred_min_production = [np.min(pred_dict_production[t]) for t in timesteps]
    pred_max_production = [np.max(pred_dict_production[t]) for t in timesteps]

    # --- Plot Consumption ---
    trace_true = go.Scatter(
        x=timesteps, y=true_flat,
        mode='lines', name='True Consumption',
        line=dict(color='rgb(30, 144, 255)', width=2)
    )
    trace_pred = go.Scatter(
        x=timesteps, y=pred_first,
        mode='lines', name='Predicted Consumption',
        line=dict(color='rgb(220, 20, 60)', width=2)
    )
    trace_interval_min = go.Scatter(
        x=timesteps, y=pred_min,
        mode='lines', line=dict(color='rgba(255,140,0,0.0)'),
        showlegend=False
    )
    trace_interval_max = go.Scatter(
        x=timesteps, y=pred_max,
        mode='lines', fill='tonexty',
        fillcolor='rgba(255,140,0,0.3)',
        line=dict(color='rgba(255,140,0,0.0)'),
        name='Prediction Interval'
    )

    fig_consumption = go.Figure(data=[
        trace_interval_min,
        trace_interval_max,
        trace_pred,
        trace_true
    ])
    fig_consumption.update_layout(
        title="True vs. Predicted Consumption with Interval",
        xaxis_title="Time Steps",
        yaxis_title="Consumption (Wh)",
        legend=dict(x=0, y=1.1, orientation='h'),
    )
    pio.write_html(fig_consumption, file=os.path.join(save_dir, "xt_consumption_interval_plotly.html"), auto_open=True)

    # --- Plot Production ---
    trace_true_p = go.Scatter(
        x=timesteps, y=true_flat_production,
        mode='lines', name='True Production',
        line=dict(color='rgb(0, 191, 255)', width=2)
    )
    trace_pred_p = go.Scatter(
        x=timesteps, y=pred_first_production,
        mode='lines', name='Predicted Production',
        line=dict(color='rgb(255, 99, 71)', width=2)
    )
    trace_interval_min_p = go.Scatter(
        x=timesteps, y=pred_min_production,
        mode='lines', line=dict(color='rgba(255,140,0,0.0)'),
        showlegend=False
    )
    trace_interval_max_p = go.Scatter(
        x=timesteps, y=pred_max_production,
        mode='lines', fill='tonexty',
        fillcolor='rgba(255,140,0,0.3)',
        line=dict(color='rgba(255,140,0,0.0)'),
        name='Prediction Interval'
    )

    fig_production = go.Figure(data=[
        trace_interval_min_p,
        trace_interval_max_p,
        trace_pred_p,
        trace_true_p
    ])
    fig_production.update_layout(
        title="True vs. Predicted Production with Interval",
        xaxis_title="Time Steps",
        yaxis_title="Production (Wh)",
        legend=dict(x=0, y=1.1, orientation='h'),
    )
    pio.write_html(fig_production, file=os.path.join(save_dir, "xt_production_interval_plotly.html"), auto_open=True)


    # Plot noise
    epsilon_true_all = [e for _, _, _, e in test_results]
    epsilon_pred_all = [e for _, _, e, _ in test_results]
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

def plot_predictions_with_uncertainty(predictions):
    import plotly.graph_objs as go
    import plotly.io as pio
    import os
    from datetime import datetime

    save_dir = os.path.join("plots", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(save_dir, exist_ok=True)

    timesteps = predictions["timesteps"]
    xt_true_all = predictions["xt_true"]
    xt_pred_all = predictions["xt_pred_mean"]
    xt_std_all = predictions["xt_pred_std"]

    # STD band (68% confidence)
    xt_upper_std = [m + s for m, s in zip(xt_pred_all, xt_std_all)]
    xt_lower_std = [m - s for m, s in zip(xt_pred_all, xt_std_all)]

    # 95% confidence band
    xt_upper_95 = [m + 1.96 * s for m, s in zip(xt_pred_all, xt_std_all)]
    xt_lower_95 = [m - 1.96 * s for m, s in zip(xt_pred_all, xt_std_all)]

    # Plotting
    trace_true = go.Scatter(
        x=timesteps, y=xt_true_all,
        mode='lines', name='True Consumption',
        line=dict(color='rgb(30, 144, 255)', width=2)
    )
    trace_pred = go.Scatter(
        x=timesteps, y=xt_pred_all,
        mode='lines', name='Predicted Mean',
        line=dict(color='rgb(220, 20, 60)', width=2)
    )

    # 95% confidence interval
    trace_band_lower_95 = go.Scatter(
        x=timesteps,
        y=xt_lower_95,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
    trace_band_upper_95 = go.Scatter(
        x=timesteps,
        y=xt_upper_95,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 140, 0, 0.15)',
        line=dict(width=0),
        name='95% Confidence Interval',
        hoverinfo='skip'
    )

    # ±1 STD interval (front)
    trace_band_lower_std = go.Scatter(
        x=timesteps,
        y=xt_lower_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    )
    trace_band_upper_std = go.Scatter(
        x=timesteps,
        y=xt_upper_std,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 140, 0, 0.3)',
        line=dict(width=0),
        name='68% Confidence Interval (±1 STD)',
        hoverinfo='skip'
    )

    fig = go.Figure(data=[
        trace_band_lower_95,
        trace_band_upper_95,
        trace_band_lower_std,
        trace_band_upper_std,
        trace_pred,
        trace_true
    ])
    fig.update_layout(
        title="Forecast with MC Dropout Uncertainty",
        xaxis_title="Time Step",
        yaxis_title="Consumption (Wh)",
        legend=dict(x=0, y=1.1, orientation='h')
    )

    pio.write_html(fig, file=os.path.join(save_dir, "xt_mc_uncertainty.html"), auto_open=True)
