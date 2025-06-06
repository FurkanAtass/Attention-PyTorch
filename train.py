import torch
from torch import nn
from model.models.transformer import Transformer
from data_utils.tokenizer import get_dataset
from torchmetrics.text import BLEUScore

from config import get_config

import pendulum
import json
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
PRINT_EVERY_N_BATCHES = 100

def get_lr_scheduler(optimizer, d_model=512, warmup_steps=4000):
    """Learning rate scheduler with warmup as described in the original paper"""
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    
    # Create the directory if it doesn't exist
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

def calculate_bleu_score(predictions, labels, target_tokenizer):
    """
    Calculate BLEU score between predictions and labels
    
    Args:
        predictions: Model predictions (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        target_tokenizer: Tokenizer to decode tokens back to text
    
    Returns:
        BLEU score
    """
    # Get predicted token ids
    _, prediction_ids = torch.max(predictions, dim=2)
    
    # Convert to lists for easier processing
    pred_ids = prediction_ids.cpu().numpy()
    label_ids = labels.cpu().numpy()
    
    # Convert token ids back to text
    pred_texts = []
    label_texts = []
    
    pad_token_id = target_tokenizer.token_to_id("[PAD]")
    eos_token_id = target_tokenizer.token_to_id("[EOS]")
    
    for i in range(len(pred_ids)):
        # Remove padding and EOS tokens, convert to text
        pred_tokens = []
        label_tokens = []
        
        for token_id in pred_ids[i]:
            if token_id != pad_token_id and token_id != eos_token_id:
                pred_tokens.append(token_id)
            else:
                break
                
        for token_id in label_ids[i]:
            if token_id != pad_token_id and token_id != eos_token_id:
                label_tokens.append(token_id)
            else:
                break
        
        # Decode tokens to text
        if len(pred_tokens) > 0:
            pred_text = target_tokenizer.decode(pred_tokens)
        else:
            pred_text = ""
            
        if len(label_tokens) > 0:
            label_text = target_tokenizer.decode(label_tokens)
        else:
            label_text = ""
            
        pred_texts.append(pred_text)
        label_texts.append([label_text])  # BLEU expects list of references
    
    # Calculate BLEU score
    bleu = BLEUScore()
    if len(pred_texts) > 0 and len(label_texts) > 0:
        return bleu(pred_texts, label_texts).item()
    else:
        return 0.0

def prepare_model_and_data(config):
    (train_dataloader, 
     valid_dataloader, 
     source_tokenizer, 
     target_tokenizer) = get_dataset(config)
    
    encoder_padding_idx = torch.tensor(source_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    decoder_padding_idx = torch.tensor(target_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)
    encoder_vocab_size = source_tokenizer.get_vocab_size()
    decoder_vocab_size = target_tokenizer.get_vocab_size()

    # Get model parameters from config if available
    model_params = config.get("model_parameters", {})

    model = Transformer(
        encoder_padding_idx,
        decoder_padding_idx,
        encoder_vocab_size,
        decoder_vocab_size,
        max_len=config["max_len"],
        device=device,
        **model_params
    )

    # Calculate adaptive warmup steps based on dataset and training configuration
    steps_per_epoch = len(train_dataloader)
    total_training_steps = steps_per_epoch * config["num_epochs"]
    
    # Use specific warmup_steps from config, or fall back to 10% of training
    warmup_steps = config.get("warmup_steps", int(0.1 * total_training_steps))
    
    print(f"Training configuration:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_training_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/steps_per_epoch:.1f} epochs)")
    print(f"  Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"  Model dropout: {model_params.get('dropout', 0.1)}")

    # Add weight decay to optimizer
    weight_decay = config.get("weight_decay", 0.0)
    base_lr = config["learning_rate"]  # Use the LR from config
    
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=base_lr,  # Use actual learning rate from config
        eps=1e-9, 
        betas=(0.9, 0.98),
        weight_decay=weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, d_model=model_params.get("d_model", 512), warmup_steps=warmup_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=decoder_padding_idx, label_smoothing=0.1)
    return train_dataloader, valid_dataloader, model, optimizer, scheduler, loss_fn, target_tokenizer

config = get_config()

(train_dataloader, 
 valid_dataloader, 
 model,
 optimizer,
 scheduler,
 loss_fn,
 target_tokenizer) = prepare_model_and_data(config)

results = get_results(config)

if results is not None and len(results) > 0:
    model_save_dir = config.get("model_save_dir", "checkpoints")
    model_path = Path(f"{model_save_dir}/model_epoch_{len(results)}.pth")
    scheduler_path = Path(f"{model_save_dir}/scheduler_epoch_{len(results)}.pth")
    
    if not model_path.exists():
        print(f"Model file {model_path} does not exist. Starting from scratch.")
        results = []
    else:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # Load scheduler state if it exists
        if scheduler_path.exists():
            print(f"Loading scheduler from {scheduler_path}")
            scheduler.load_state_dict(torch.load(scheduler_path))
        else:
            print("No scheduler state found, calculating correct scheduler step...")
            # Calculate total steps completed so far
            steps_per_epoch = len(train_dataloader)
            total_steps_completed = len(results) * steps_per_epoch
            
            print(f"Advancing scheduler by {total_steps_completed} steps to resume at correct LR")
            
            # Manually advance scheduler to correct position
            for step in range(total_steps_completed):
                scheduler.step()
                if (step + 1) % 1000 == 0:  # Progress indicator
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Step {step + 1}/{total_steps_completed}, LR: {current_lr:.6f}")
            
            final_lr = scheduler.get_last_lr()[0]
            print(f"Scheduler resumed at step {total_steps_completed}, LR: {final_lr:.6f}")
            
    print(f"Loaded model from epoch {len(results)}")

model.to(device)

for epoch in range(len(results), config["num_epochs"]):
    avg_train_loss = 0.
    avg_validation_loss = 0.
    avg_bleu = 0.
    epoch_start_time = pendulum.now()

    model.train()
    for i, batch in enumerate(train_dataloader):

        enc_input = batch["encoder_input"].to(device)
        dec_input = batch["decoder_input"].to(device)
        label = batch["decoder_output"].to(device)

        prediction = model(enc_input, dec_input)

        optimizer.zero_grad()

        loss = loss_fn(prediction.permute(0, 2, 1), label)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        avg_train_loss += loss.item()

        if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
                  f"Batch {i + 1}/{len(train_dataloader)} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Avg Train Loss: {avg_train_loss / (i + 1):.4f} | "
                  f"LR: {current_lr:.6f}")
            
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

            batch_bleu = calculate_bleu_score(prediction, label, target_tokenizer)
            avg_bleu += batch_bleu

            if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
                print(f"Validation Batch {i + 1}/{len(valid_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"BLEU Score: {batch_bleu:.4f}")

    avg_validation_loss /= len(valid_dataloader)
    avg_bleu /= len(valid_dataloader)
    epoch_end_time = pendulum.now()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Validation Loss: {avg_validation_loss:.4f} | "
          f"Validation BLEU Score: {avg_bleu:.4f} | "
          f"Time: {(epoch_end_time - epoch_start_time).in_words()} ")
    
    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_validation_loss,
        "validation_bleu_score": avg_bleu,
        "learning_rate": current_lr,
        "time": (epoch_end_time - epoch_start_time).in_words()
    }
    results.append(epoch_results)
    save_results(config, results)
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(config["model_save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), f"{config["model_save_dir"]}/model_epoch_{epoch + 1}.pth")
    torch.save(scheduler.state_dict(), f"{config["model_save_dir"]}/scheduler_epoch_{epoch + 1}.pth")



