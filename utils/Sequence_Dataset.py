from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, data_tensor, seq_length):
        self.pairs = create_input_target_pairs(data_tensor, seq_length)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return x.float(), y.float()

def create_input_target_pairs(data_tensor, seq_length):
    input_target_pairs = []
    for i in range(len(data_tensor) - seq_length):
        x = data_tensor[i:i + seq_length]
        y = data_tensor[i + 1:i + seq_length + 1]
        input_target_pairs.append((x, y))
    return input_target_pairs