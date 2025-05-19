import torch
from torch import nn
from layers.multihead_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.attention1 = MultiHeadAttention(num_heads, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.attention2 = MultiHeadAttention(num_heads, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attention1 = self.attention1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1)
        add_norm1 = self.norm1(x + attention1)

        # Second attention layer
        attention2 = self.attention2(enc_output, enc_output, add_norm1, padding_mask)
        attention2 = self.dropout2(attention2)
        add_norm2 = self.norm2(add_norm1 + attention2)

        # Feed forward network
        ffn = self.ffn(add_norm2)
        ffn = self.dropout2(ffn)
        add_norm3 = self.norm3(add_norm2 + ffn)

        return add_norm3
