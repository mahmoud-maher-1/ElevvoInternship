import streamlit as st
from transformers import pipeline, AutoTokenizer, TFAutoModelForQuestionAnswering # <-- Changed import

# --- App Configuration ---
st.set_page_config(
    page_title="QA Model Showcase",
    page_icon="🤖",
    layout="wide",
)

# --- Model Loading ---

# Use Streamlit's caching to load models only once.
@st.cache_resource
def load_qa_pipeline(model_path):
    """Loads a Question Answering pipeline from a local path for TensorFlow."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use TFAutoModelForQuestionAnswering for TensorFlow models
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_path) # <-- Changed model class
        # The pipeline will automatically use the CPU if no GPU is found
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer) # <-- Removed device=-1
        return qa_pipeline
    except Exception as e:
        # Return the exception to be handled in the main app
        return e

# Define paths to your models
MODEL_PATHS = {
    "BERT": "./models/my_first_QA_model_bert_base",
    "ALBERT": "./models/my_first_QA_model_albert_base",
    "RoBERTa": "./models/my_first_QA_model_roberta_base",
}

# Load all models and store them in a dictionary
with st.spinner("Loading models... This may take a moment."):
    pipelines = {}
    for model_name, path in MODEL_PATHS.items():
        loaded_pipeline = load_qa_pipeline(path)
        if isinstance(loaded_pipeline, Exception):
            st.error(f"Failed to load {model_name} model from {path}. Error: {loaded_pipeline}")
            continue
        pipelines[model_name] = loaded_pipeline

# Check if any models were loaded successfully
if not pipelines:
    st.error("No models could be loaded. Please check the model paths and files. The application will now stop.")
    st.stop()

# --- App UI ---

st.title("📄 Question Answering with Transformer Models")
st.write(
    "Select a model, provide a passage of text (the context), and ask a question. "
    "The model will find the answer within the text."
)

# Model selection dropdown
available_models = list(pipelines.keys())
selected_model_name = st.selectbox(
    "Choose a Model", options=available_models
)

# Get the corresponding pipeline for the selected model
qa_pipeline = pipelines[selected_model_name]

st.header("1. Provide the Context")
default_context = "The James Webb Space Telescope is the largest optical telescope in space. Its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope. It was launched by an Ariane 5 rocket from Kourou, French Guiana, in 2021."
context = st.text_area(
    "Paste the passage of text here:",
    height=200,
    value=default_context
)

st.header("2. Ask a Question")
question = st.text_input(
    "Ask a question based on the context above:",
    value="When was the James Webb telescope launched?"
)

# --- Prediction and Output ---

if st.button("Find the Answer"):
    if context and question:
        with st.spinner(f"Asking {selected_model_name}..."):
            try:
                result = qa_pipeline(question=question, context=context)
                st.subheader("💡 Answer")
                st.info(f"**Answer:** {result['answer']}")
                st.write(f"**Confidence Score:** {result['score']:.4f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please provide both a context and a question.")

# --- Footer ---
st.markdown("---")
st.markdown("App built with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face](https://huggingface.co/).")