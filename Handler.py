import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from DataSet import DataLoader
from model import LSTM_Model, GRU_Model
import numpy as np
import os
import math


torch.manual_seed(1111)


class Handler:
    def __init__(self, model, model_name, data_path, train_batch_size, eval_batch_size, sequence_length, criterion,
                 epochs, lr, dropout, device):
        self.model = model
        self.model_name = model_name
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sequence_length = sequence_length
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.lr_tracker = []
        self.dropout = dropout
        self.device = device
        self.dataloader = DataLoader(data_path=data_path, train_batch_size=train_batch_size,
                                     eval_batch_size=eval_batch_size, sequence_length=sequence_length)

        self.model.to(device)

    def train_one_epoch(self, epoch):
        self.model.train()

        train_dataloader = self.dataloader.train

        total_loss = 0
        avg_loss = 0
        avg_perplexity = 0
        print_every = 200

        hidden = self.model.init_hidden(self.train_batch_size, self.device)
        for batch_idx, i in enumerate(range(0, train_dataloader.size(0) - 1, self.sequence_length)):
            data, target = self.dataloader.get_batch(train_dataloader, i)

            data = data.to(self.device)
            target = target.to(self.device)

            hidden = self.model.detach_hidden(hidden, self.device)

            self.model.zero_grad()

            result, hidden = self.model(data, hidden)
            loss = self.criterion(result, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            for p in self.model.parameters():
                p.data.add_(p.grad.data, alpha=-self.lr)

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            avg_perplexity = math.exp(avg_loss)

            if batch_idx % print_every == 0 and batch_idx > 0:
                print(f'Epoch[{epoch + 1:03d}] | {batch_idx} / {len(train_dataloader)// self.sequence_length} '
                      f'batches | Loss: {avg_loss:.2f} | lr {self.lr:.2f} | Perplexity: {avg_perplexity:.2f}')

        return avg_loss, avg_perplexity

    def evaluate_model(self, dataloader, data_type):
        self.model.eval()

        total_loss = 0
        avg_loss = 0
        if data_type == "Train":
            hidden = self.model.init_hidden(self.train_batch_size, self.device)
        else:
            hidden = self.model.init_hidden(self.eval_batch_size, self.device)
        for batch_idx, i in enumerate(range(0, dataloader.size(0) - 1, self.sequence_length)):
            with torch.no_grad():
                data, target = self.dataloader.get_batch(dataloader, i)

                data = data.to(self.device)
                target = target.to(self.device)

                result, hidden = self.model(data, hidden)

                loss = self.criterion(result, target)

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

        return avg_loss, math.exp(avg_loss)

    def save_data(self, average_train_perplexity, average_validation_perplexity, average_test_perplexity):
        plt.plot(average_train_perplexity, label="Train Perplexity")
        plt.plot(average_test_perplexity, label="Test Perplexity")
        plt.title(f"{self.model_name}: lr {self.lr:.4f} Dropout {self.dropout}")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend(loc="upper right")
        plt.xticks(np.arange(0, len(average_train_perplexity), 5))
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), "saved_models", self.model_name, "Convergence Graph.png"))
        plt.close()

        plt.plot(self.lr_tracker)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("lr")
        plt.xticks(np.arange(0, len(self.lr_tracker), 5))
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), "saved_models", self.model_name, "LR Graph.png"))
        plt.close()

        train_perplexity_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name,
                                                 "Train Perplexity Values.log")
        test_perplexity_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name,
                                                "Test Perplexity Values.log")
        validation_perplexity_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name,
                                                      "Validation Perplexity Values.log")
        train_file = open(train_perplexity_filepath, "w")
        test_file = open(test_perplexity_filepath, "w")
        validation_file = open(validation_perplexity_filepath, "w")

        for i in range(len(average_train_perplexity)):
            if i != len(average_train_perplexity) - 1:
                train_file.write(str(average_train_perplexity[i]) + ", ")
                test_file.write(str(average_test_perplexity[i]) + ", ")
                validation_file.write(str(average_validation_perplexity[i]) + ", ")
            else:
                train_file.write(str(average_train_perplexity[i]))
                test_file.write(str(average_test_perplexity[i]))
                validation_file.write(str(average_validation_perplexity[i]))
        train_file.write(f'\nLowest Perplexity: {min(average_train_perplexity)}')
        test_file.write(f'\nLowest Perplexity: {min(average_test_perplexity)}')
        validation_file.write(f'\nLowest Perplexity: {min(average_validation_perplexity)}')

        train_file.close()
        test_file.close()
        validation_file.close()

    def run(self):
        print(f"Start Training {self.model_name}.")
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models"))
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name))

        best_loss = torch.inf
        checkpoint_filename = os.path.join(os.getcwd(), "saved_models", self.model_name, f'{self.model_name}.pt')

        train_perplexity = []
        test_perplexity = []
        validation_perplexity = []
        for epoch in range(self.epochs):
            start = time.time()
            self.lr_tracker.append(self.lr)
            print("-" * 80)
            self.train_one_epoch(epoch)

            train_avg_loss, train_avg_perplexity = self.evaluate_model(self.dataloader.train, "Train")
            validation_avg_loss, validation_avg_perplexity = self.evaluate_model(self.dataloader.valid, "Validation")
            test_avg_loss, test_avg_perplexity = self.evaluate_model(self.dataloader.test, "Test")
            print("-" * 80)
            end = time.time()
            print(f"| End of epoch {epoch + 1} | Epoch RunTime {(end - start):.2f}s | Train ppl "
                  f"{train_avg_perplexity:.2f} | Valid ppl {validation_avg_perplexity:.2f} | Test ppl "
                  f"{test_avg_perplexity:.2f}")

            train_perplexity.append(train_avg_perplexity)
            test_perplexity.append(test_avg_perplexity)
            validation_perplexity.append(validation_avg_perplexity)

            if validation_avg_loss < best_loss:
                print(f"Saving Checkpoint {checkpoint_filename}")
                state = {
                    "model": self.model.state_dict()
                }
                torch.save(state, checkpoint_filename)
                best_loss = validation_avg_loss
            else:
                self.lr /= 4.0

        self.save_data(train_perplexity, validation_perplexity, test_perplexity)

    def load_model(self):
        print(f"Loading Model {self.model_name}.")
        model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
        print("Loaded Model Successfully.")

    def test(self):
        print(f"Start Testing {self.model_name}.")
        train_avg_loss, train_avg_perplexity = self.evaluate_model(self.dataloader.train, "Train")
        validation_avg_loss, validation_avg_perplexity = self.evaluate_model(self.dataloader.valid, "Validation")
        test_avg_loss, test_avg_perplexity = self.evaluate_model(self.dataloader.test, "Test")

        print(f"\nTrain Average Perplexity {train_avg_perplexity:.2f}.")
        print(f"Validation Average Perplexity {validation_avg_perplexity:.2f}.")
        print(f"Test Average Perplexity {test_avg_perplexity:.2f}.")


if __name__ == "__main__":
    dropout = 0
    input_size = 1
    hidden_size = 200
    num_layers = 2
    num_tokens = 10000
    num_embeddings = 200
    sequence_length = 35
    criterion = nn.CrossEntropyLoss()
    epochs = 40
    train_batch_size = 20
    eval_batch_size = 10
    lr = 20
    model_name = "LSTM_Dropout_0"
    data_path = "./data"

    model = LSTM_Model(hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout,
                       num_tokens=num_tokens,
                       num_embeddings=num_embeddings)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    handler = Handler(model=model,
                      model_name=model_name,
                      data_path=data_path,
                      train_batch_size=train_batch_size,
                      eval_batch_size=eval_batch_size,
                      sequence_length=sequence_length,
                      criterion=criterion,
                      epochs=epochs,
                      lr=lr,
                      dropout=dropout,
                      device=device)
    handler.run()
