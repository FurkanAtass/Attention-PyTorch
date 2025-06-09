# Attention-PyTorch

A PyTorch implementation of the Transformer architecture for neural machine translation, based on the seminal paper "Attention Is All You Need" by Vaswani et al.

## Overview

This project implements a complete Transformer model from scratch using PyTorch, designed for sequence-to-sequence tasks like machine translation.

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
- **Resume Training**: Automatic checkpoint and scheduler loading

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
