import torch
from torch import nn
from blocks.decoder_block import DecoderBlock
from embedding.input_embedding import InputEmbedding
from embedding.positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers, max_len, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)
        ])


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # x: (batch_size, T)
        x = self.input_embedding(x)
        # x: (batch_size, T, d_model)
        x = self.positional_encoding(x)
        # x: (batch_size, T, d_model)
        for block in self.decoder_blocks:
            x = block(x, enc_output, look_ahead_mask, padding_mask)
            # x: (batch_size, T, d_model)
        return x