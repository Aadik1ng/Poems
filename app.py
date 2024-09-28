import streamlit as st
from scripts.generate_poem import generate_poem
from scripts.utils import load_config
from scripts.generate_sonnets import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load configuration
config = load_config()

# Streamlit UI
st.title('Poem n Sonnets')
st.write('Generate poems or sonnets using pre-trained GPT-2 models.')

# Function to create a tooltip
def create_tooltip(label, tooltip):
    return f"""
    <div style="position: relative; display: inline-block;">
        <span>{label}</span>
        <span class="tooltiptext" style="visibility: hidden; width: 200px; background-color: black; color: #fff; text-align: center; border-radius: 6px; padding: 5px 0; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s;">
            {tooltip}
        </span>
    </div>
    <style>
    div:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """

# Select generation type
gen_type = st.selectbox('Choose text generation type:', ['Poem', 'Sonnet'])

# Input prompt
prompt = st.text_input('Enter a prompt for the text:', '')

# Parameters for text generation with tooltips
if gen_type == 'Poem' or gen_type == 'Sonnet':
    max_length = st.slider('Max Length:', min_value=10, max_value=1000, value=100, step=10, help='Maximum length of the generated text.')
    st.markdown(create_tooltip('', 'Maximum length of the generated text.'), unsafe_allow_html=True)

    num_return_sequences = st.number_input('Number of Return Sequences:', min_value=1, max_value=10, value=1, help='Number of different sequences to return.')
    st.markdown(create_tooltip('', 'Number of different sequences to return.'), unsafe_allow_html=True)

    temperature = st.slider('Temperature:', min_value=0.1, max_value=1.0, value=0.7, step=0.1, help='Controls the randomness: lower is less random, higher is more random.')
    st.markdown(create_tooltip('', 'Controls the randomness: lower is less random, higher is more random.'), unsafe_allow_html=True)

    top_p = st.slider('Top P (Nucleus Sampling):', min_value=0.1, max_value=1.0, value=0.9, step=0.1, help='Limits to the top p probability mass (nucleus sampling).')
    st.markdown(create_tooltip('', 'Limits to the top p probability mass (nucleus sampling).'), unsafe_allow_html=True)

    top_k = st.number_input('Top K (Sampling):', min_value=1, max_value=100, value=50, help='Limits the sampling to the top k tokens.')
    st.markdown(create_tooltip('', 'Limits the sampling to the top k tokens.'), unsafe_allow_html=True)

    repetition_penalty = st.slider('Repetition Penalty:', min_value=1.0, max_value=5.0, value=2.0, step=0.1, help='Penalty for repeating the same token.')
    st.markdown(create_tooltip('', 'Penalty for repeating the same token.'), unsafe_allow_html=True)

# Generate button
if st.button('Generate Text'):
    if prompt:
        if gen_type == 'Poem':
            with st.spinner('Generating poem...'):
                model_path = config['model']['save_path']
                poem = generate_poem(
                    prompt, 
                    model_path, 
                    max_length=max_length, 
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                )
                st.success('Poem generated!')
                st.text_area('Generated Poem', poem, height=200)
        elif gen_type == 'Sonnet':
            with st.spinner('Generating sonnet...'):
                model_save_path = config['model']['save_path2']
                generated_text = generate_text(
                    GPT2LMHeadModel.from_pretrained(model_save_path).to('cuda'), 
                    GPT2Tokenizer.from_pretrained(model_save_path), 
                    prompt, 
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences
                )
                st.success('Sonnet generated!')
                st.text_area('Generated Sonnet', generated_text, height=200)
    else:
        st.warning('Please enter a prompt.')

# Running Streamlit
if __name__ == "__main__":
    st.write("Please run this script using the Streamlit CLI.")
