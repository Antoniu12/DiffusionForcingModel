import matplotlib

# Try TkAgg first (recommended)
matplotlib.use("TkAgg")  # Most compatible with PyCharm

# If TkAgg doesn't work, use Qt5Agg
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt


class TrainingPlotter:
    def __init__(self):
        """Initialize storage for training metrics."""
        self.mse_loss_list = []
        self.r2_score_list = []
        self.smape_score_list = []
        self.epsilon_pred_list = []
        self.epsilon_true_list = []

    def update_metrics(self, mse, r2, smape, epsilon_pred, epsilon_true):
        """
        Store the metrics and predictions after each epoch.
        """
        self.mse_loss_list.append(mse)
        self.r2_score_list.append(r2)
        self.smape_score_list.append(smape)

        # Store only a subset of epsilon values to prevent excessive memory usage
        if len(self.epsilon_pred_list) < 10:
            self.epsilon_pred_list.append(epsilon_pred.detach().cpu().numpy())
            self.epsilon_true_list.append(epsilon_true.detach().cpu().numpy())

    def plot_metrics(self):
        """
        Plot MSE, R² Score, and SMAPE over epochs inside PyCharm SciView.
        """
        epochs = range(1, len(self.mse_loss_list) + 1)

        plt.figure(figsize=(12, 4))
        plt.plot(epochs, self.mse_loss_list, label="MSE Loss", color='blue', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Training MSE Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()  # ✅ Fix: Fully renders inside PyCharm SciView

        plt.figure(figsize=(12, 4))
        plt.plot(epochs, self.r2_score_list, label="R² Score", color='green', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("R² Score")
        plt.title("Training R² Score Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(epochs, self.smape_score_list, label="SMAPE (%)", color='red', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("SMAPE (%)")
        plt.title("Training SMAPE Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_predictions(true_values, predicted_values):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True Consumption", color='blue', linewidth=2)
    plt.plot(predicted_values, label="Predicted Consumption", color='red', linestyle='dashed')
    plt.xlabel("Time Steps")
    plt.ylabel("Energy Consumption (Wh)")
    plt.title("True vs. Predicted Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()