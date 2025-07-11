
class Main:
    def __init__(self):
        print("Initialization Main...")
        self.seq_length = 24
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        print("1. Train diffusion on H-dataset")
        print("2. Train diffusion on AEP-dataset")
        print("3. Train LSTM & Transformer on all datasets")
        print("4. Generate test dataset predictions for diffusion next-hour")
        print("5. Generate test dataset predictions for LSTM and Transformer next-hour")
        print("6. Generate day predictions Autoregressive Diffusion")
        print("7. Generate day predictions Autoregressive LSTM and Transformer")

        opt = input("Pick an option (1, 2, 3, 4, 5, 6, 7): ")
        if opt == "1":
            idx = input("Input index H-dataset (1-20): ")
            train_diffusion_on_h_dataset(int(idx))
        elif opt == "2":
            train_diffusion_on_aep_dataset()
        elif opt == "3":
            tain_lstm_and_transformer()
        elif opt == "4":
            show_plot_for_diffusion()
        elif opt == "5":
            m = input("LSTM or Transformer?: ")
            show_plot_for_LT(m)
        elif opt == "6":
            autoregressive_diffusion()
        elif opt == "7":
            m = input("LSTM or Transformer?: ")
            autoregressive_LT(m)
        else:
            print("Unknown option!")

if __name__ == "__main__":
    import torch
    from DataPreprocessing.training import (train_diffusion_on_h_dataset,
                                            train_diffusion_on_aep_dataset,
                                            tain_lstm_and_transformer, show_plot_for_diffusion, show_plot_for_LT,
                                            autoregressive_LT, autoregressive_diffusion)

    main = Main()
    main.run()
