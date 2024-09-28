import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scripts.utils import load_config

def generate_poem(prompt, model_path, max_length=100, num_return_sequences=1, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=2.0):
    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Set the pad token ID to eos_token_id to avoid warnings
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode the prompt with attention mask
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    # Generate the poem text with adjustable parameters
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,  # Adjust max length
        num_return_sequences=num_return_sequences,
        temperature=temperature,  # Adjust temperature
        top_p=top_p,        # Adjust nucleus sampling
        top_k=top_k,         # Adjust top_k sampling
        repetition_penalty=repetition_penalty,  # Adjust repetition penalty
        do_sample=True    # Enable sampling
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    model_path = config['model']['save_path']  # Update with the path to your saved model
    prompt = "The autumn leaves"  # Change the prompt as needed

    poem = generate_poem(prompt, model_path)
    print(poem)
