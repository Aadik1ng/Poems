
# Poem and Sonnet Generator

## Project Overview

The **Poem and Sonnet Generator** is a machine learning project that leverages the power of the GPT-2 model, a variant of the Transformer architecture developed by OpenAI. The aim is to generate creative and coherent poems and sonnets based on custom training data.

## Key Components of the Project

1. **Model Training**: The project involves training a GPT-2 model on custom datasets, specifically designed to teach the model the structure and style of poems and sonnets.
2. **Text Generation**: Once trained, the model can generate new text based on user-provided prompts. This can be done through a command-line script or a user-friendly web interface using Streamlit.
3. **User Interface**: The Streamlit application provides an interactive interface where users can input prompts and generate poems and sonnets with various parameters.

## Detailed Project Structure

- **config/config.yaml**: Configuration file containing paths to datasets and model parameters.
- **models/**: Directory where trained models for poems and sonnets are saved.
- **scripts/**: Directory containing the scripts for generating text, training models, and utility functions.
- **app.py**: Streamlit application for generating poems and sonnets interactively.
- **README.md**: Documentation for the project.

## How GPT-2 is Used

### 1. Training the Model

To train the GPT-2 model on custom datasets, the following steps are followed:

1. **Data Preparation**: Custom datasets are collected and prepared. These datasets contain examples of poems and sonnets that the model will learn from. The datasets should be formatted in a way that GPT-2 can understand, typically as plain text files with one poem or sonnet per line.

2. **Configuration**: The `config.yaml` file is set up to specify the paths to these datasets and other training parameters such as the number of epochs, batch size, and save paths for the models.

    ```yaml
    model:
      name: "gpt2"
      save_path: "./models/poems"
      save_path2: "./models/sonnets"

    data:
      form: "data/form_poems.txt"
      topic: "data/topic_poems.txt"
      shakespeare: "data/shakespeare.txt"

    training:
      output_dir: "./models"
      epochs: 3
      batch_size: 4
      save_steps: 10
      max_steps: 1000
    ```

3. **Training Process**: The training process is executed on a local machine with a 6GB VRAM GPU. The training typically takes around 30 minutes.

    ```python
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10,
        max_steps=1000,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
    )

    trainer.train()
    ```

### 2. Generating Text

Once the model is trained, you can generate new poems and sonnets using the following steps:

    ```python
    # Load the trained model and tokenizer
    model_name = "./models/poems"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Generate text
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```

## Conclusion

This project demonstrates the capabilities of fine-tuning the GPT-2 model for generating creative text. The provided scripts and configuration files make it straightforward to replicate and extend the training and text generation process.

## Getting Started

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Aadik1ng/Poem-Generator.git
    cd Poem-Generator
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare datasets**: Place your datasets in the `data` directory.

4. **Train the model**:
    ```sh
    python scripts/train.py
    ```

5. **Generate text**:
    ```sh
    python scripts/generate_text.py --prompt "Your prompt here"
    ```

6. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

Happy poem and sonnet generating!
    