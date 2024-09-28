import os
import yaml
from transformers import TextDataset
def load_config() :
    with open('config\config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        return config
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )