import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import torchtext.data.utils as ttdutils

from text_dataset import TextDataset


def get_dataset_and_vocab(dataset_name, seq_len=32, line_slice=None):
    """Returns a dataset and a vocab for the given dataset name"""
    print(dataset_name)
    if dataset_name == "wikitext2":
        dataset, vocab = get_wikitext2(seq_len, line_slice)
    elif dataset_name == "jokes":
        dataset, vocab = get_jokes(seq_len=seq_len, line_slice=line_slice)
    elif dataset_name == "minimal":
        dataset, vocab = get_minimal(seq_len)
    else:
        raise ValueError("Dataset not found")
    return dataset, vocab
    

def get_wikitext2(seq_len, line_slice=None):
    train_iter = WikiText2(root="./data", split="train")

    if line_slice is not None:
        train_iter = list(train_iter)[:line_slice]

    tokenizer = ttdutils.get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    data = [torch.LongTensor([vocab(tokenizer(item))]) for item in train_iter]
    data = tuple(filter(lambda x: x.numel() > 0, data))
    data = torch.cat(data, dim=1).squeeze(0)

    return TextDataset(data, seq_len=seq_len), vocab



def get_jokes(seq_len, line_slice=None):
    df = pd.read_csv("resources/jokes.csv")
    train_iter = df["Joke"].values
    if line_slice is not None:
        train_iter = list(train_iter)[:line_slice]
    
    # train_iter = list(train_iter)[:dataset_slice]
    tokenizer = ttdutils.get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    data = [torch.LongTensor([vocab(tokenizer(item))]) for item in train_iter]
    data = tuple(filter(lambda x: x.numel() > 0, data))
    data = torch.cat(data, dim=1).squeeze(0)
  
    
    return TextDataset(data, seq_len=seq_len), vocab

def get_minimal(seq_len):
    train_iter = ["I am sentient and you are not", "The world we live in is a simulation"]
    tokenizer = ttdutils.get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    data = [torch.LongTensor([vocab(tokenizer(item))]) for item in train_iter]
    data = tuple(filter(lambda x: x.numel() > 0, data))
    data = torch.cat(data, dim=1).squeeze(0)

    return TextDataset(data, seq_len=seq_len), vocab


if __name__ == "__main__":
    dataset_name = "wikitext2"
    dataset, vocab = get_dataset_and_vocab(dataset_name, line_slice=100)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        print(batch)
        print(batch.shape)
        break