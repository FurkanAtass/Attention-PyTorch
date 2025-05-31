import torch
from torch import nn
from model.models.transformer import Transformer
from data_utils.tokenizer import get_dataset

from config import get_config

import pendulum
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
PRINT_EVERY_N_BATCHES = 100

def prepare_model_and_data(config):
    (train_dataloader, 
     valid_dataloader, 
     source_tokenizer, 
     target_tokenizer) = get_dataset(config)
    
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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=encoder_padding_idx, label_smoothing=0.1)
    return train_dataloader, valid_dataloader, model, optimizer, loss_fn

config = get_config()

(train_dataloader, 
 valid_dataloader, 
 model,
 optimizer,
 loss_fn) = prepare_model_and_data(config)

model.to(device)
torch.save(model.state_dict(), "model.pth")
print(f"Train dataloader length: {len(train_dataloader)}")
print(f"Validation dataloader length: {len(valid_dataloader)}")


def accuracy(predictions, labels):
    _, prediction_classes = torch.max(predictions, dim=2)
    correct = (prediction_classes == labels).float()
    acc = correct.sum() / correct.numel()
    return acc.item()
  
# save loss acc and model
for epoch in range(config["num_epochs"]):
    avg_train_loss = 0.
    avg_validation_loss = 0.
    avg_acc = 0.
    epoch_start_time = pendulum.now()

    model.train()
    for i, batch in enumerate(train_dataloader):

        enc_input = batch["encoder_input"].to(device)
        dec_input = batch["decoder_input"].to(device)
        label = batch["decoder_output"].to(device)

        prediction = model(enc_input, dec_input)

        optimizer.zero_grad()

        # loss = loss_fn(prediction.view(-1, model.decoder_vocab_size), label.view(-1))
        loss = loss_fn(prediction.permute(0, 2, 1), label)

        loss.backward()

        optimizer.step()

        avg_train_loss += loss.item()

        if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
                  f"Batch {i + 1}/{len(train_dataloader)} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Avg Train Loss: {avg_train_loss / (i + 1):.4f}")
        # if i == 100:
        #     break 
    avg_train_loss /= len(train_dataloader)

    model.eval()
    for i, batch in enumerate(valid_dataloader):
        enc_input = batch["encoder_input"].to(device)
        dec_input = batch["decoder_input"].to(device)
        label = batch["decoder_output"].to(device)

        with torch.inference_mode():
            prediction = model(enc_input, dec_input)

            loss = loss_fn(prediction.permute(0, 2, 1), label)

            avg_validation_loss += loss.item()

            batch_acc = accuracy(prediction, label)
            avg_acc += batch_acc

            if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
                print(f"Validation Batch {i + 1}/{len(valid_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Accuracy: {batch_acc:.4f}")

    avg_validation_loss /= len(valid_dataloader)
    avg_acc /= len(valid_dataloader)
    epoch_end_time = pendulum.now()
    print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Validation Loss: {avg_validation_loss:.4f} | "
          f"Validation Accuracy: {avg_acc:.4f} | "
          f"Time: {(epoch_end_time - epoch_start_time).in_words()} ")
