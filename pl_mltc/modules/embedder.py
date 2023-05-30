from typing import Optional

import torch
from torch import nn
import numpy as np

from pl_mltc.data.vocab import Vocab


class Embedder(nn.Module):
    def __init__(self, filename: Optional[str], n_dim: int, vocab: Vocab) -> None:
        super().__init__()
        embeddings = np.random.randn(len(vocab), n_dim)

        if filename is not None:
            with open(filename) as fh:
                header = fh.readline()
                n_pretrained, pretrained_dim = header.strip().split()
                n_pretrained = int(n_pretrained)
                pretrained_dim = int(pretrained_dim)
                assert n_dim == pretrained_dim, f"n_dim (`{n_dim}`) should equal to pretrained_dim (`{pretrained_dim}`)."
                for line in fh:
                    line = line.strip().split()
                    word = line[0]
                    word_index = vocab.word2id(word)
                    if word_index != vocab.oov_index:
                        vector = [float(_) for _ in line[1:]]
                        embeddings[word_index] = vector

        self.embedding = nn.Embedding(len(vocab), n_dim)
        embeddings[vocab.pad_index] = 0.0
        self.embedding.weight.data = torch.Tensor(embeddings)
    
    def forward(self, word_indices):
        return self.embedding(word_indices)
