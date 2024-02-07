import os
import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_path = os.path.join(os.getcwd(), path)
        self.word2idx = {}  # word: index
        self.idx2word = []  # position(index): word
        self.tokenize()

    def tokenize(self):
        with open(self.data_path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if word not in self.word2idx:
                        self.idx2word.append(word)
                        self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        with open(self.data_path, 'r') as file:
            line_count = sum(1 for line in file)
        return line_count

    def __getitem__(self, idx):
        output = []
        with open(self.data_path, 'r') as file:
            line = file.readline()
            line = line.split()
            for word in line:
                output.append(self.word2idx[word])
            output.append(self.word2idx['<eos>'])
        output = torch.unsqueeze(torch.Tensor(output), dim=1)
        return output


if __name__ == "__main__":
    data_path = "./data/ptb.train.txt"
    dataset = DataSet(data_path)
    print("Loaded Data!")
    print(f"Dataset Length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample: {sample}")
    print(f"Sample Shape: {sample.shape}")

