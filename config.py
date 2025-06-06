def get_config():
    return {
    "dataset_name": "Helsinki-NLP/opus_books",
    "dataset_config": "de-en",
    "source_language": "en",
    "target_language": "de",

    "train_split": 0.9,
    
    "tokenizer_path": {
        "source": "tokenizers/source_tokenizer",
        "target": "tokenizers/target_tokenizer"
    },
    "results_path": "results.json",
    "model_save_dir": "checkpoints",
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "max_len": 500,
    
    "weight_decay": 1e-4,
    "warmup_steps": 4000,
    
    "model_parameters": {
        "d_model": 512,
        "num_heads": 8,
        "dff": 2048,
        "num_layers": 6,
        "dropout": 0.2
    }
}
