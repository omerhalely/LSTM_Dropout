import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        return h, c


class GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size=1, hidden_size=200, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        pass

    def init_hidden(self):
        pass


if __name__ == "__main__":
    input_size = 1
    hidden_size = 200
    num_layers = 2
    dropout = 0.5
    batch_size = 20

    lstm_model = LSTM_Model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    x = torch.rand(batch_size, 25, input_size)
    hidden = lstm_model.init_hidden(batch_size=batch_size)
    output, hidden = lstm_model(x, hidden)

    print(f"Output Shape: {output.shape}")
    print(f"Hidden State Shape: {hidden[0].shape}")
    print(f"Cell State Shape: {hidden[1].shape}")