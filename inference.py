import torch
from model.models.transformer import Transformer
from config import get_config
from data_utils.tokenizer import get_or_build_tokenizer
from torchmetrics.text import BLEUScore
from model.models.transformer import Transformer

device = "cuda" if torch.cuda.is_available() else "cpu"

def inference(sentence, model: Transformer, source_tokenizer, target_tokenizer, max_len, device):
    
    source_sos_token = torch.tensor(source_tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
    source_eos_token = torch.tensor(source_tokenizer.token_to_id("[EOS]"), dtype=torch.int64)

    target_sos_token = torch.tensor(target_tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
    target_eos_token = torch.tensor(target_tokenizer.token_to_id("[EOS]"), dtype=torch.int64)
    
    input_ids = source_tokenizer.encode(sentence).ids

    encoder_input = torch.cat([
        source_sos_token.unsqueeze(0),
        torch.tensor(input_ids, dtype=torch.int64),
        source_eos_token.unsqueeze(0)
    ]).unsqueeze(0).to(device)

    decoder_input = target_sos_token.unsqueeze(0).unsqueeze(0).to(device)
    
    model.to(device)
    model.eval()

    with torch.inference_mode():
        enc_padding_mask = model.enc_mask(encoder_input)

        encoder_output = model.encoder(encoder_input, enc_padding_mask)

        for _ in range(max_len):
            decoder_padding_mask = model.dec_mask(decoder_input)

            decoder_output = model.decoder(
                decoder_input, 
                encoder_output, 
                decoder_padding_mask, 
                enc_padding_mask
            )

            final_output = model.final_layer(decoder_output)
            next_token = torch.argmax(final_output[:, -1, :], dim=-1)

            if next_token.item() == target_eos_token.item():
                break

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

        generated_tokens = decoder_input.squeeze(0)[1:].cpu().numpy()  # Remove SOS token
        generated_text = target_tokenizer.decode(generated_tokens.tolist())
        print(generated_text)

        return generated_text


# Load actual tokenizers
config = get_config()
source_tokenizer = get_or_build_tokenizer(config, None, None, "source")
target_tokenizer = get_or_build_tokenizer(config, None, None, "target")


encoder_padding_idx = torch.tensor(source_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
decoder_padding_idx = torch.tensor(target_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
encoder_vocab_size = source_tokenizer.get_vocab_size()
decoder_vocab_size = target_tokenizer.get_vocab_size()

model = Transformer(
    encoder_padding_idx,
        decoder_padding_idx,
        encoder_vocab_size,
        decoder_vocab_size,
        max_len=config["max_len"],
        device=device
)

model.load_state_dict(torch.load(f"{config["model_save_dir"]}/model_epoch_28.pth"))
input_sentence = input("Enter a sentence in English: ")

output = inference(
    input_sentence,model, source_tokenizer, target_tokenizer, config["max_len"], "cuda"
)






