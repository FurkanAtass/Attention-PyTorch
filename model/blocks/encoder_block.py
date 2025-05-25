import torch
from torch import nn
from model.layers.multihead_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model)
        )

    def forward(self, x, padding_mask=None):
        attention = self.attention(x, x, x, padding_mask)
        attention = self.dropout(attention)
        add_norm1 = self.norm1(x + attention)

        ffn = self.ffn(add_norm1)
        add_norm2 = self.norm2(add_norm1 + ffn)

        return add_norm2

