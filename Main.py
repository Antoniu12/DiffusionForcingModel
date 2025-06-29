from datetime import datetime

class Main:
    def __init__(self):
        print("Initializare Main...")
        self.seq_length = 24
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        print("1. Train diffusion pe H-dataset")
        print("2. Train diffusion pe AEP-dataset")
        print("3. Train LSTM & Transformer pe toate seturile")
        opt = input("Alege opțiunea (1/2/3): ")
        if opt == "1":
            idx = input("Introdu index H-dataset (1-20): ")
            train_diffusion_on_h_dataset(int(idx))
        elif opt == "2":
            train_diffusion_on_aep_dataset()
        elif opt == "3":
            tain_lstm_and_transformer()
        elif opt == "4":
            show_plot_for_diffusion()
        elif opt == "5":
            m = input("LSTM sau Transformer?")
            show_plot_for_LT(m)
        else:
            print("Opțiune necunoscută.")

if __name__ == "__main__":
    import torch
    from DataPreprocessing.training import (train_diffusion_on_h_dataset,
                                            train_diffusion_on_aep_dataset,
                                            tain_lstm_and_transformer, show_plot_for_diffusion, show_plot_for_LT)

    main = Main()
    main.run()
