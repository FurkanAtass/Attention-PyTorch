import torch

from model.models.transformer import Transformer
from data_utils.tokenizer import get_dataset

from config import get_config

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
        max_len=config["max_len"]
    )

    return train_dataloader, valid_dataloader, model

config = get_config()
train_dataloader, valid_dataloader, model = prepare_model_and_data(config)

batch = next(iter(train_dataloader))

enc_input = batch["encoder_input"]
dec_input = batch["decoder_input"]

print(f"enc_input shape: {enc_input.shape}")
print(f"dec_input shape: {dec_input.shape}")
output = model(enc_input, dec_input)
print(f"output shape: {output.shape}")