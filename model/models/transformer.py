import torch
from torch import nn

from model.models.encoder import Encoder
from model.models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            encoder_padding_idx,
            decoder_padding_idx,
            encoder_vocab_size, 
            decoder_vocab_size,
            d_model=512, 
            num_heads=8, 
            dff=2048, 
            num_layers=6, 
            max_len=10000, 
            dropout=0.1
        ):
        super(Transformer, self).__init__()

        self.encoder_padding_idx = encoder_padding_idx
        self.decoder_padding_idx = decoder_padding_idx
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.encoder = Encoder(
            vocab_size=encoder_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=decoder_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.final_layer = nn.Linear(d_model, decoder_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, dec_input):
        # enc_input: (batch_size, src_len)
        # dec_input: (batch_size, tgt_len)

        enc_padding_mask = self.enc_mask(enc_input)
        dec_padding_mask, look_ahead_mask = self.dec_mask(dec_input)

        enc_output = self.encoder(enc_input, enc_padding_mask)
        dec_output = self.decoder(dec_input, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)
        final_output = self.dropout(final_output)

        return final_output

    def enc_mask(self, enc_input):
        enc_padding_mask = (enc_input != self.encoder_padding_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return enc_padding_mask
    
    def dec_mask(self, dec_input):
        # dec_input: (batch_size, T)
        dec_padding_mask = (dec_input != self.decoder_padding_idx).unsqueeze(1).unsqueeze(2)
        look_ahead_mask = torch.triu(torch.ones((dec_input.size(1), dec_input.size(1))), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)
        return dec_padding_mask, look_ahead_mask

    


