import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import torchtext.data.utils as ttdutils

class TextDataset(Dataset):
    def __init__(self, data, seq_len=32):
        super(TextDataset, self).__init__()
        self.seq_len = seq_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]