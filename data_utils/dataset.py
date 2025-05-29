import torch
from torch import nn
from torch.utils.data import Dataset

class TranslateDataset(Dataset):
    def __init__(
            self, 
            dataset, 
            source_tokenizer, 
            target_tokenizer, 
            source_language, 
            target_language, 
            max_len 
    ):
        super().__init__()
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        self.source_language = source_language
        self.target_language = target_language

        self.max_len = max_len
        
        self.source_sos_token = torch.Tensor(source_tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
        self.source_eos_token = torch.Tensor(source_tokenizer.token_to_id("[EOS]"), dtype=torch.int64)
        self.source_pad_token = torch.Tensor(source_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)

        self.target_sos_token = torch.Tensor(target_tokenizer.token_to_id("[SOS]"), dtype=torch.int64)
        self.target_eos_token = torch.Tensor(target_tokenizer.token_to_id("[EOS]"), dtype=torch.int64)
        self.target_pad_token = torch.Tensor(target_tokenizer.token_to_id("[PAD]"), dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]

        source_data = data["translation"][self.source_language]
        target_data = data["translation"][self.target_language]

        encoder_input_tokens = self.source_tokenizer.encode(source_data).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_data).ids

        encoder_num_padding_tokens = self.max_len - len(encoder_input_tokens) - 2
        decoder_num_padding_tokens = self.max_len - len(decoder_input_tokens) - 1

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence longer than max_len")
        
        encoder_input = torch.cat([
            self.source_eos_token,
            encoder_input_tokens,
            self.source_eos_token,
            torch.Tensor([self.source_pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.target_sos_token,
            decoder_input_tokens,
            torch.Tensor([self.target_pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ])

        decoder_output = torch.cat([
            decoder_input_tokens,
            self.target_eos_token,
            torch.Tensor([self.target_pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.max_len
        assert decoder_input.size(0) == self.max_len
        assert decoder_output.size(0) == self.max_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_output": decoder_output
        }

