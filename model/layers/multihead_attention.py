import torch
from torch import nn
from self_attention import SelfAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = SelfAttention()
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(q.size(0), q.size(1), self.num_heads, -1).transpose(1, 2) 
        # before transpose: (batch_size, T, num_heads, d_model/num_heads) 
        # after transpose: (batch_size, num_heads, T, d_model/num_heads)

        k = k.view(k.size(0), k.size(1), self.num_heads, -1).transpose(1, 2)

        v = v.view(v.size(0), v.size(1), self.num_heads, -1).transpose(1, 2)

        attention = self.attention(q, k, v, mask) # shape: (batch_size, num_heads, T, d_model/num_heads)

        #concatenate the heads

        attention = attention.transpose(1, 2).contiguous() # (batch_size, T, num_heads, d_model/num_heads)
        attention = attention.view(attention.size(0), attention.size(1), -1) # (batch_size, T, d_model)

        attention = self.output_linear(attention) # (batch_size, T, d_model)

        return attention





