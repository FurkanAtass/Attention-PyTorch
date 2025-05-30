import torch
from model.models.transformer import Transformer

x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])

trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

model = Transformer(
    encoder_padding_idx=src_pad_idx,
    decoder_padding_idx=trg_pad_idx,
    encoder_vocab_size=src_vocab_size,
    decoder_vocab_size=trg_vocab_size,
)

out = model(x, trg[:, :-1])

print(out.shape) 