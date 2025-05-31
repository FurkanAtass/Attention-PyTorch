import torch
from torch import nn
from model.models.transformer import Transformer
from data_utils.tokenizer import get_dataset

from config import get_config

import pendulum
import json
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
PRINT_EVERY_N_BATCHES = 100

def get_results(config):
    results_file = config.get("results_path", "result.json")
    model_save_dir = config.get("model_save_dir", "checkpoints")

    results_path = Path(f"{model_save_dir}/{results_file}")
    if not Path(results_path).exists():
        return []
    with open(results_path, "r") as f:
        results = json.load(f)
        print(f"Loaded results from {results_path}")
        print(f"Results: {results}")
    return results

def save_results(config, results):
    results_file = config.get("results_path", "results.json")
    model_save_dir = config.get("model_save_dir", "checkpoints")

    results_path = Path(f"{model_save_dir}/{results_file}")
    with open(results_path, "w") as f:
        json.dump(results, f)

def accuracy(predictions, labels):
    _, prediction_classes = torch.max(predictions, dim=2)
    correct = (prediction_classes == labels).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

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

results = get_results(config)

if results is not None and len(results) > 0:
    model_save_dir = config.get("model_save_dir", "checkpoints")
    model_path = Path(f"{model_save_dir}/model_epoch_{len(results)}.pth")
    if not model_path.exists():
        print(f"Model file {model_path} does not exist. Starting from scratch.")
        results = []
    else:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from epoch {len(results)}")

model.to(device)

for epoch in range(len(results), config["num_epochs"]):
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
    
    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_validation_loss,
        "validation_accuracy": avg_acc,
        "time": (epoch_end_time - epoch_start_time).in_words()
    }
    results.append(epoch_results)
    save_results(config, results)
    torch.save(model.state_dict(), f"{config["model_save_dir"]}/model_epoch_{epoch + 1}.pth")


