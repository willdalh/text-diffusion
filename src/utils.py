import torch

def text_collate_fn(batch):
    """Combine batch list of tensors so that batch_size is the second dimension"""
    return torch.stack(batch, dim=1)


def read_glove_vectors(glove_path, vocab, embed_dim):
    """Read in GloVe embeddings for words in vocab"""
    covered_words = []
    embedding = torch.zeros(len(vocab), embed_dim)
    with open(glove_path, "r", encoding="utf-8") as f:
        glove_vectors = {}
        for line in f:
            line = line.split()
            word = line[0]
            if word in vocab:
                index = vocab[word]
                embedding[index] = torch.Tensor([float(x) for x in line[1:]])
                covered_words.append(word)

    print(f"Found {len(covered_words)} out of {len(vocab)} words in GloVe, meaning {len(vocab) - len(covered_words)} words are missing.")
    for word in vocab.get_stoi():
        if word not in covered_words:
            index = vocab[word]
            embedding[index] = torch.randn(embed_dim)
            
    return embedding