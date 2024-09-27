import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_poem(prompt, model_path):
    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Set the pad token ID to eos_token_id to avoid warnings
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode the prompt with attention mask
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    # Generate the poem text with repetition penalty and top_k sampling
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=100,  # Increase max length for longer output
        num_return_sequences=1,
        temperature=0.7,  # Lower temperature for less randomness
        top_p=0.9,        # Use nucleus sampling
        top_k=50,         # Limit to top 50 tokens
        repetition_penalty=2.0,  # Penalty to reduce repetition
        do_sample=True    # Enable sampling
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_path = config['model']['save_path']  # Update with the path to your saved model
    prompt = "The autumn leaves"  # Change the prompt as needed

    poem = generate_poem(prompt, model_path)
    print(poem)
