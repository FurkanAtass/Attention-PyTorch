from torch import nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d):
        super(InputEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.d = d
        self.embedding = nn.Embedding(vocab_size, d)

    def forward(self, x):
        x = self.embedding(x) / math.sqrt(self.d) # shape: (batch_size, T, d)

        return x