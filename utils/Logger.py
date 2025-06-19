import os
from datetime import datetime

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "training_log.txt")
        os.makedirs(self.save_dir, exist_ok=True)

        with open(self.log_file, "w") as f:
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.metrics_history = {
            'epoch': [],
            'mse': [],
            'r2_xt': [],
            'r2_epsilon': [],
            'smape': [],
        }

    def log_info(self, text):
        print(text)
        with open(self.log_file, "a") as f:
            f.write(text + "\n")

    def log_model_config(self, model_name, config_dict):
        self.log_info(f"Model: {model_name}")
        self.log_info("Model Configuration:")
        for key, value in config_dict.items():
            self.log_info(f"  {key}: {value}")
        self.log_info("\n")
    def save_final_metrics(self, crps_score):
        self.log_info("\nFinal Evaluation Metrics:")
        self.log_info(f"CRPS Score: {crps_score:.6f}")