import os
import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import load_config, load_dataset

def main():
    config = load_config()

    model_name = config['model']['name']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load datasets for poems and sonnets
    poem_dataset = load_dataset(config['data']['form'], tokenizer)
    sonnet_dataset = load_dataset(config['data']['shakespeare'], tokenizer)
    
    # Define output directories for models
    base_output_dir = 'models'
    poem_output_dir = os.path.join(base_output_dir, 'poems')
    sonnet_output_dir = os.path.join(base_output_dir, 'sonnets')
    
    # Create directories if they don't exist
    os.makedirs(poem_output_dir, exist_ok=True)
    os.makedirs(sonnet_output_dir, exist_ok=True)

    # Check if GPU is available and use it if so
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Training for Poem Model
    poem_training_args = TrainingArguments(
        output_dir=poem_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        save_steps=config['training']['save_steps'],
        max_steps=config['training']['max_steps'],
    )

    poem_trainer = Trainer(
        model=model,
        args=poem_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=poem_dataset,
    )

    print("Training Poem Model on", device + "...")
    poem_trainer.train()
    poem_trainer.save_model(poem_output_dir)
    tokenizer.save_pretrained(poem_output_dir)

    # Training for Sonnet Model
    sonnet_training_args = TrainingArguments(
        output_dir=sonnet_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        save_steps=config['training']['save_steps'],
        max_steps=config['training']['max_steps'],
    )

    sonnet_trainer = Trainer(
        model=model,
        args=sonnet_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=sonnet_dataset,
    )

    print("Training Sonnet Model on", device + "...")
    sonnet_trainer.train()
    sonnet_trainer.save_model(sonnet_output_dir)
    tokenizer.save_pretrained(sonnet_output_dir)

if __name__ == "__main__":
    main()
