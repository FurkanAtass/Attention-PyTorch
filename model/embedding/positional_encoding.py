import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.d = d
        self.max_len = max_len

        self.encoding = torch.zeros(max_len, d) # shape: (max_len, d)

        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)) # shape: (d/2)

        # For even indices, apply sin
        self.encoding[:, 0::2] = torch.sin(position * div_term) # shape: (max_len, d/2)
        
        # For odd indices, apply cos
        self.encoding[:, 1::2] = torch.cos(position * div_term) # shape: (max_len, d/2)

        self.encoding = self.encoding.unsqueeze(0) # shape: (1, max_len, d)

        self.register_buffer('positional_encoding', self.encoding)
    def forward(self, x):
        # x: (batch_size, T, d)
        # Use the encoding for the first T positions
        x = x + self.encoding[:, :x.size(1), :].requires_grad(False) # shape: (batch_size, T, d)
        x = self.dropout(x)

        return x