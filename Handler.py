import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from DataSet import DataSet
from model import LSTM_Model, GRU_Model


class Handler:
    def __init__(self, model, model_name, data_path, optimizer, epochs, lr, batch_size, device):
        self.model = model
        self.model_name = model_name
        self.data_path = data_path
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.train_dataset = DataSet(path=data_path)

        self.model.to(device)

    def train_one_epoch(self, epoch):
        self.model.train()

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        print_every = len(train_dataloader) // 10
        hidden = self.model.init_hidden(self.batch_size)
        for batch_idx, sentence in enumerate(train_dataloader):
            sentence = sentence.to(self.device)
            hidden = hidden.to(device)

            result, hidden = self.model(sentence, hidden)


if __name__ == "__main__":
    dropout = 0.5
    input_size = 1
    hidden_size = 200
    num_layers = 2
    model = LSTM_Model(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout)
    model_name = "LSTM_Dropout"
    data_path = "./data/ptb.train.txt"
    lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    epochs = 10
    batch_size = 20
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    handler = Handler(model=model,
                      model_name=model_name,
                      data_path=data_path,
                      optimizer=optimizer,
                      epochs=epochs,
                      lr=lr,
                      batch_size=batch_size,
                      device=device
                      )
    handler.train_one_epoch(0)
