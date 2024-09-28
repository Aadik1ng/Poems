import os
import yaml
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import load_config

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=2.0, num_return_sequences=1):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Explicitly declare paths
    config = load_config()
    model_save_path = config['model']['save_path2']
    prompt = "To be, or not to be, that is the question:"

    # Load configuration
    model_name = config['model']['name']
    model = GPT2LMHeadModel.from_pretrained(model_save_path).to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, max_length=150)
    
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
