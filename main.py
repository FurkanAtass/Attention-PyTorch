import torch
from model.models.transformer import Transformer
from train import calculate_bleu_score
from config import get_config
from data_utils.tokenizer import get_or_build_tokenizer


# Test the transformer and BLEU score
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

# Load actual tokenizers
config = get_config()
source_tokenizer = get_or_build_tokenizer(config, None, None, "source")
target_tokenizer = get_or_build_tokenizer(config, None, None, "targer")

encoder_padding_idx = torch.tensor(source_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
decoder_padding_idx = torch.tensor(target_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
encoder_vocab_size = source_tokenizer.get_vocab_size()
decoder_vocab_size = target_tokenizer.get_vocab_size()

model = Transformer(
    encoder_padding_idx,
        decoder_padding_idx,
        encoder_vocab_size,
        decoder_vocab_size,
)

# Get model output
with torch.inference_mode():
    out = model(x, trg[:, :-1])
print(f"Model output shape: {out.shape}")

# Test BLEU score calculation
target_labels = trg[:, 1:]  # Target without SOS token
bleu_score = calculate_bleu_score(out, target_labels, target_tokenizer)

print(f"BLEU Score: {bleu_score:.4f}")

# Let's also print some decoded examples to see what's happening
_, prediction_ids = torch.max(out, dim=2)
print("\nExample predictions vs targets:")
for i in range(min(2, out.size(0))):  # Print first 2 examples
    pred_tokens = prediction_ids[i].cpu().numpy()
    target_tokens = target_labels[i].cpu().numpy()
    
    # Remove padding tokens for display
    pad_token_id = target_tokenizer.token_to_id("[PAD]")
    pred_clean = [t for t in pred_tokens if t != pad_token_id][:5]  # Show first 5 non-pad tokens
    target_clean = [t for t in target_tokens if t != pad_token_id][:5]
    
    pred_text = target_tokenizer.decode(pred_clean)
    target_text = target_tokenizer.decode(target_clean)
    
    print(f"Sample {i+1}:")
    print(f"  Predicted: {pred_text}")
    print(f"  Target:    {target_text}")
    print(f"  Pred IDs:  {pred_clean}")
    print(f"  Target IDs: {target_clean}")
    print() 