import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from data_utils.dataset import TranslateDataset

def get_all_sentences(dataset, language):
    for data in dataset:
        yield data['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_path"])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens = ["UNK", "PAD", "SOS", "EOS"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)

        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    raw_dataset = load_dataset(config["dataset_name"], config["dataset_config"], split="train")
    source_tokenizer = get_or_build_tokenizer(config, raw_dataset, config["source_language"])
    target_tokenizer = get_or_build_tokenizer(config, raw_dataset, config=["target_language"])

    train_dataset_size = int(len(raw_dataset) * config["train_split"])
    valid_dataset_size = len(raw_dataset) - train_dataset_size

    raw_train_dataset, raw_valid_dataset = random_split(raw_dataset, [train_dataset_size, valid_dataset_size])
    
    source_max_len = 0
    target_max_len = 0

    for item in raw_dataset:
        source_len = source_tokenizer.encode(item["translation"][config["source_language"]])
        target_len = target_tokenizer.encode(item["translation"][config["target_language"]])

        source_max_len = max(source_max_len, source_len)
        target_max_len = max(target_max_len, target_len)

    train_dataset = TranslateDataset(
        raw_train_dataset,
        source_tokenizer,
        target_tokenizer,
        config["source_language"],
        config["target_language"],
        source_max_len
    )

    valid_dataset = TranslateDataset(
        raw_valid_dataset,
        source_tokenizer,
        target_tokenizer,
        config["source_language"],
        config["target_language"],
        target_max_len
    )

    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, config["batch_size"])

    return train_dataloader, valid_dataloader, source_tokenizer, target_tokenizer
