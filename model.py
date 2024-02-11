import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_tokens, num_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_embeddings)
        self.lstm = nn.LSTM(input_size=num_embeddings, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, num_tokens)

    def forward(self, x, hidden):
        x = self.drop(self.encoder(x))
        output, hidden = self.lstm(x, hidden)
        output = self.drop(output)
        decoded_output = self.decoder(output.view((output.size(0) * output.size(1), output.size(2))))
        decoded_output = decoded_output.view((output.size(0), output.size(1), decoded_output.size(1)))
        return decoded_output, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h, c


class GRU_Model(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_tokens, num_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_embeddings)
        self.gru = nn.GRU(input_size=num_embeddings, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, num_tokens)

    def forward(self, x, hidden):
        x = self.drop(self.encoder(x))
        output, hidden = self.gru(x, hidden)
        output = self.drop(output)
        decoded_output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        decoded_output = decoded_output.view((output.size(0), output.size(1), decoded_output.size(1)))
        return decoded_output, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h


if __name__ == "__main__":
    input_size = 1
    hidden_size = 200
    num_layers = 2
    dropout = 0.5
    batch_size = 20
    num_tokens = 10000
    num_embeddings = 256
    sequence_length = 35

    lstm_model = LSTM_Model(hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            num_tokens=num_tokens,
                            num_embeddings=num_embeddings
                            )

    gru_model = GRU_Model(hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          num_tokens=num_tokens,
                          num_embeddings=num_embeddings)

    x = torch.randint(0, 10000, (sequence_length, batch_size))
    lstm_hidden = lstm_model.init_hidden(batch_size=batch_size)
    lstm_output, lstm_hidden = lstm_model(x, lstm_hidden)

    print("LSTM")
    print(f"Output Shape: {lstm_output.shape}")
    print(f"Hidden State Shape: {lstm_hidden[0].shape}")
    print(f"Cell State Shape: {lstm_hidden[1].shape}")
    print()

    gru_hidden = gru_model.init_hidden(batch_size=batch_size)
    gru_output, gru_hidden = gru_model(x, gru_hidden)

    print("GRU")
    print(f"Output Shape: {gru_output.shape}")
    print(f"Hidden State Shape: {gru_hidden.shape}")


