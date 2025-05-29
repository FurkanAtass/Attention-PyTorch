import torch
from torch import nn
from model.embedding.input_embedding import InputEmbedding
from model.embedding.positional_encoding import PositionalEncoding
from model.blocks.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers, max_len, dropout=0.1, device="cpu"):
        super(Encoder, self).__init__()

        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout, device)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        # x: (batch_size, T)
        x = self.input_embedding(x)
        # x: (batch_size, T, d_model)
        x = self.positional_encoding(x)
        # x: (batch_size, T, d_model)
        for block in self.encoder_blocks:
            x = block(x, padding_mask)
            # x: (batch_size, T, d_model)
        return x

