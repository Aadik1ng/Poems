# Poem and Sonnet Generator

## Overview
This project allows you to generate poems and sonnets using pre-trained GPT-2 models. The project includes scripts for training the models and a Streamlit application for generating and displaying the texts.

## Project Structure
\`\`\`plaintext
├── config
│   └── config.yaml
├── models
│   ├── poems
│   └── sonnets
├── scripts
│   ├── generate.py
│   ├── model_training.py
│   └── utils.py
├── app.py
└── README.md
\`\`\`

## Requirements
- Python 3.7+
- PyTorch
- Transformers
- Streamlit
- PyYAML

## Installation
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/Aadik1ng/poem-sonnet-generator.git
   cd poem-sonnet-generator
   \`\`\`

2. Create a virtual environment and activate it:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
   \`\`\`

3. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Ensure you have the necessary data files in the \`data\` directory as specified in \`config/config.yaml\`.

## Configuration
Update the \`config/config.yaml\` file with the appropriate paths and parameters:
\`\`\`yaml
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
\`\`\`

## Training the Models
Run the following command to train the models for poems and sonnets:
\`\`\`bash
python scripts/model_training.py
\`\`\`

## Generating Texts
You can generate poems and sonnets using the \`generate.py\` script or the Streamlit app.

### Using \`generate.py\`
Edit the \`generate.py\` script to change the prompts and run:
\`\`\`bash
python scripts/generate.py
\`\`\`

### Using the Streamlit App
Run the Streamlit app:
\`\`\`bash
streamlit run app.py
\`\`\`

Use the web interface to input prompts and adjust parameters for generating poems and sonnets.

## Scripts Description

### \`scripts/generate.py\`
This script generates text based on prompts and the pre-trained models.

### \`scripts/model_training.py\`
This script trains the GPT-2 models on the provided datasets.

### \`scripts/utils.py\`
Utility functions for loading configurations and datasets.

### \`app.py\`
Streamlit application for generating poems and sonnets using a web interface.



## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

---

This README file provides a comprehensive guide to setting up, training, and using the Poem and Sonnet Generator project. For any further questions, feel free to reach out or open an issue on the GitHub repository.
