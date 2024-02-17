import os
import torch
from torch.autograd import Variable


class Dictionary(object):
    """Build word2idx and idx2word from Corpus(train/val/test)"""
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        """Create/Update word2idx and idx2word"""
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class DataLoader(object):
    """Corpus Tokenizer"""
    def __init__(self, data_path, train_batch_size, eval_batch_size, sequence_length):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sequence_length = sequence_length
        self.dictionary = Dictionary()
        self.train = self.batchify(self.tokenize(os.path.join(data_path, 'ptb.train.txt')), self.train_batch_size)
        self.valid = self.batchify(self.tokenize(os.path.join(data_path, 'ptb.valid.txt')), self.eval_batch_size)
        self.test = self.batchify(self.tokenize(os.path.join(data_path, 'ptb.test.txt')), self.eval_batch_size)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # line to list of token + eos
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def batchify(self, data, batch_size):
        number_of_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, number_of_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data

    def get_batch(self, source, i):
        seq_len = min(self.sequence_length, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len])
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target


if __name__ == "__main__":
    data_path = "./data"
    train_batch_size = 20
    eval_batch_size = 20
    sequence_length = 35
    dataset = DataLoader(data_path=data_path,
                         train_batch_size=train_batch_size,
                         eval_batch_size=eval_batch_size,
                         sequence_length=sequence_length)

