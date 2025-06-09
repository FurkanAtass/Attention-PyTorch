# Attention-PyTorch

A PyTorch implementation of the Transformer architecture for neural machine translation, based on the seminal paper "Attention Is All You Need" by Vaswani et al.

## Overview

This project implements a complete Transformer model from scratch using PyTorch, designed for sequence-to-sequence tasks like machine translation. The implementation includes:

- **Full Transformer Architecture**: Encoder-decoder model with multi-head self-attention
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Pre-norm architecture
- **Feed-Forward Networks**: Position-wise feed-forward layers
- **Custom Training Loop**: With learning rate scheduling and warmup
- **BLEU Score Evaluation**: Automatic evaluation metrics
- **Tokenization**: Custom tokenizer integration

## Features

- ✅ **Complete Transformer Implementation** - Full encoder-decoder architecture
- ✅ **Configurable Hyperparameters** - Easy configuration management
- ✅ **Multi-head Attention** - Scalable attention mechanism
- ✅ **Learning Rate Scheduling** - Warmup and decay as per original paper
- ✅ **BLEU Score Metrics** - Automatic translation quality evaluation
- ✅ **Resume Training** - Checkpoint saving and loading
- ✅ **GPU Support** - CUDA acceleration
- ✅ **Inference Mode** - Ready-to-use translation inference

## Project Structure

```
Attention-PyTorch/
├── model/
│   ├── models/
│   │   ├── transformer.py      # Main Transformer model
│   │   ├── encoder.py          # Encoder implementation
│   │   └── decoder.py          # Decoder implementation
│   ├── blocks/                 # Transformer blocks (attention, FFN)
│   ├── layers/                 # Individual layers
│   └── embedding/              # Embedding and positional encoding
├── data_utils/
│   ├── dataset.py              # Dataset loading and preprocessing
│   └── tokenizer.py            # Tokenization utilities
├── checkpoints/                # Model checkpoints
├── tokenizers/                 # Saved tokenizer files
├── config.py                   # Configuration settings
├── train.py                    # Training script
├── inference.py                # Inference script
└── README.md
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Attention-PyTorch.git
cd Attention-PyTorch
```

2. **Install dependencies:**
```bash
pip install torch torchmetrics datasets tokenizers pendulum pathlib
```

3. **Create required directories:**
```bash
mkdir -p checkpoints tokenizers
```

## Configuration

The model configuration is managed through `config.py`. Key parameters include:

```python
{
    "dataset_name": "Helsinki-NLP/opus_books",    # HuggingFace dataset
    "dataset_config": "de-en",                    # Language pair
    "source_language": "en",                      # Source language
    "target_language": "de",                      # Target language
    
    "batch_size": 8,                              # Training batch size
    "num_epochs": 50,                             # Number of training epochs
    "max_len": 500,                               # Maximum sequence length
    
    "model_parameters": {
        "d_model": 512,                           # Model dimension
        "num_heads": 8,                           # Number of attention heads
        "dff": 2048,                              # Feed-forward dimension
        "num_layers": 6,                          # Number of encoder/decoder layers
        "dropout": 0.1                            # Dropout rate
    }
}
```

## Usage

### Training

To train the Transformer model:

```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Build source and target tokenizers
- Initialize the Transformer model
- Train with learning rate scheduling
- Save checkpoints every epoch
- Calculate BLEU scores for evaluation

### Inference

To translate text using a trained model:

```bash
python inference.py
```

The script will prompt you to enter English text and will output the German translation.

**Example:**
```
Enter a sentence in English: Hello, how are you?
Output: Hallo, wie geht es dir?
```

### Programmatic Inference

You can also use the inference function programmatically:

```python
from inference import inference, model, source_tokenizer, target_tokenizer
from config import get_config

config = get_config()
result = inference(
    "Your text here", 
    model, 
    source_tokenizer, 
    target_tokenizer, 
    config["max_len"], 
    "cuda"
)
```

## Model Architecture

The implementation follows the original Transformer architecture:

### Encoder
- **Multi-Head Self-Attention**: Parallel attention heads
- **Position-wise Feed-Forward**: Two linear transformations
- **Residual Connections**: Skip connections around sub-layers
- **Layer Normalization**: Applied before each sub-layer

### Decoder
- **Masked Multi-Head Self-Attention**: Prevents future token access
- **Encoder-Decoder Attention**: Attends to encoder outputs
- **Position-wise Feed-Forward**: Same as encoder
- **Residual Connections & Layer Norm**: Same as encoder

### Key Components
- **Positional Encoding**: Sinusoidal embeddings for sequence position
- **Multi-Head Attention**: Scaled dot-product attention mechanism
- **Feed-Forward Networks**: Position-wise fully connected layers

## Training Details

### Learning Rate Scheduling
The model uses the learning rate schedule from the original paper:
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

### Optimization
- **Optimizer**: Adam with β₁=0.9, β₂=0.98, ε=1e-9
- **Warmup Steps**: 4000 steps (configurable)
- **Weight Decay**: L2 regularization
- **Label Smoothing**: 0.1 for cross-entropy loss

### Evaluation
- **BLEU Score**: Calculated on validation set
- **Checkpoint Saving**: Model and scheduler states saved each epoch
- **Resume Training**: Automatic checkpoint loading

## Performance

The model achieves competitive BLEU scores on the German-English translation task:
- Training converges in ~30-50 epochs
- BLEU scores typically reach 10+ on validation set

## TODO

Future improvements to make the implementation more faithful to the original paper:

- [ ] **WMT 2014 Dataset**: Replace current dataset with WMT 2014 English-German translation dataset as used in the original "Attention Is All You Need" paper
- [ ] **BPE Tokenizer**: Implement Byte-Pair Encoding (BPE) tokenization instead of the current tokenizer
  - Use 37,000 merge operations for source and target vocabularies
  - Shared vocabulary between English and German as in the original paper
- [ ] **Training Configuration**: Match exact hyperparameters from the original paper:
  - 100,000 training steps
  - Base model: 6 layers, 512 d_model, 8 heads, 2048 d_ff
  - Big model: 6 layers, 1024 d_model, 16 heads, 4096 d_ff
- [ ] **Evaluation**: Use the same evaluation metrics and test sets as the original paper
- [ ] **Multi-GPU Training**: Implement distributed training for larger models and datasets
