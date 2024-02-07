import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=200, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden


class GRU_Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=200, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        pass

if __name__ == "__main__":
    drop_out = 0.5

    lstm_model = LSTM_Model(dropout=drop_out)
    gru_model = GRU_Model(dropout=drop_out)
