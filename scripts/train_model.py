import os
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    form_dataset = load_dataset(config['data']['form'], tokenizer)
    topic_dataset = load_dataset(config['data']['topic'], tokenizer)

    combined_dataset = form_dataset + topic_dataset

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        save_steps=config['training']['save_steps'],
        max_steps=config['training']['max_steps'],
    )

    trainer = Trainer(
        model=model.to('cuda'),
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=combined_dataset,
    )

    trainer.train()
    trainer.save_model(config['model']['save_path'])
    tokenizer.save_pretrained(config['model']['save_path'])  # Save the tokenizer

if __name__ == "__main__":
    main()
