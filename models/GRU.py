from torch import nn


class GRUWithLayerNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    def forward(self, x, h=None):
        # x = self.norm(x)
        out, h = self.gru(x, h)
        return out, h