import os
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from utils import load_config,load_dataset



def main():
    config = load_config()

    model_name = config['model']['name']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    form_dataset = load_dataset(config['data']['form'], tokenizer)
    topic_dataset = load_dataset(config['data']['topic'], tokenizer)
    
    combined_dataset = form_dataset + topic_dataset
    #for sonnets uncomment the below and comment the above
    #shakespeare_dataset = load_dataset('data\shakespeare.txt',tokenizer)
    if not combined_dataset:#+
        output_dir='williams'
    else: 
        output_dir='poems'
    
    training_args = TrainingArguments(
        output_dir="./results/f'{output_dir}'",
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
        train_dataset=combined_dataset,#shakespeare_dataset
    )

    trainer.train()
    if(combined_dataset):
        trainer.save_model(config['model']['save_path'])
        tokenizer.save_pretrained(config['model']['save_path']) 
    else:
        trainer.save_model(config['model']['save_path2'])
        tokenizer.save_pretrained(config['model']['save_path2'])  



if __name__ == "__main__":
    main()