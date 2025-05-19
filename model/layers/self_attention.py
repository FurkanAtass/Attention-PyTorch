import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(-1) 

    def forward(self, q, k, v, mask = None):

        batch_size, num_heads, T, d = q.size()

        attention = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf')) # values 0 in mask are set to -inf

        attention = self.softmax(attention) # softmax to get attention weights q * k^T

        attention = torch.matmul(attention, v) # shape: (batch_size, num_heads, T, d)
        
        return attention
    
