import torch
from torch import nn
from model.models.transformer import Transformer
from data_utils.tokenizer import get_dataset

from config import get_config

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def prepare_model_and_data(config):
    (train_dataloader, 
     valid_dataloader, 
     source_tokenizer, 
     target_tokenizer) = get_dataset(config)
    
    encoder_padding_idx = torch.tensor(source_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    decoder_padding_idx = torch.tensor(target_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    encoder_vocab_size = source_tokenizer.get_vocab_size()
    decoder_vocab_size = target_tokenizer.get_vocab_size()

    print(f"Encoder vocab size: {encoder_vocab_size}")
    print(f"Decoder vocab size: {decoder_vocab_size}")
    model = Transformer(
        encoder_padding_idx,
        decoder_padding_idx,
        encoder_vocab_size,
        decoder_vocab_size,
        max_len=config["max_len"],
        device=device
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=encoder_padding_idx, label_smoothing=0.1)
    return train_dataloader, valid_dataloader, model, optimizer, loss_fn

config = get_config()

(train_dataloader, 
 valid_dataloader, 
 model,
 optimizer,
 loss_fn) = prepare_model_and_data(config)

model.train()
model.to("mps")
batch = next(iter(train_dataloader))

enc_input: torch.Tensor = batch["encoder_input"].to("mps")
dec_input = batch["decoder_input"].to("mps")

print(f"enc_input shape: {enc_input.device}")
print(f"dec_input shape: {dec_input}")
output = model(enc_input, dec_input)
print(f"output shape: {output.shape}")