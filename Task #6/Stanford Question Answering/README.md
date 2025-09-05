# Documentation for Training and Deployment of Transformer QA Models

## Overview

This documentation provides a comprehensive explanation of the workflow
for training and deploying three transformer-based Question Answering
(QA) models: **BERT**, **ALBERT**, and **RoBERTa**. The workflow is
divided into two parts across two files:

1.  **Training File (`stanford-question-answering.ipynb`)**\
    This Jupyter notebook contains the end-to-end process of preparing
    the Stanford Question Answering Dataset (SQuAD), preprocessing the
    data, fine-tuning the models, and saving them locally.

2.  **Deployment File (`app.py`)**\
    This Streamlit application showcases the trained models, allowing
    users to interact with them by providing a context passage and
    asking questions. The application loads the models and exposes a
    user-friendly interface for inference.

------------------------------------------------------------------------

## Part 1: Training the Models (`stanford-question-answering.ipynb`)

### 1. Dataset Preparation

-   The notebook uses the **Stanford Question Answering Dataset
    (SQuAD)**, a benchmark dataset for extractive question answering
    tasks.
-   Each record in SQuAD consists of a **context passage**, a
    **question**, and the **answer span** (start and end positions
    within the context).

### 2. Preprocessing

-   Tokenization is performed using Hugging Face's `AutoTokenizer` for
    each respective model (BERT, ALBERT, RoBERTa).\
-   The preprocessing ensures:
    -   Alignment of input IDs, attention masks, and token type IDs (for
        models requiring them).\
    -   Correct mapping of answer spans to token indices for supervised
        training.

### 3. Model Training

-   Three pretrained models were fine-tuned for the QA task:

    -   **BERT (bert-base-uncased)**\
    -   **ALBERT (albert-base-v2)**\
    -   **RoBERTa (roberta-base)**

-   Each model is wrapped with `TFAutoModelForQuestionAnswering` to
    adapt them to extractive QA.

-   Training involves:

    -   Optimizers such as Adam or AdamW.\
    -   Loss function specific to span prediction.\
    -   Batch training with GPU acceleration (if available).

### 4. Saving Trained Models

-   After training, each model is saved locally in separate directories:
    -   `./models/my_first_QA_model_bert_base`
    -   `./models/my_first_QA_model_albert_base`
    -   `./models/my_first_QA_model_roberta_base`
-   These directories contain both the model weights and tokenizer
    configurations.

------------------------------------------------------------------------

## Part 2: Deployment with Streamlit (`app.py`)

### 1. Application Setup

-   The app is powered by **Streamlit**, a Python library for building
    interactive web apps.\
-   The configuration includes page title, icon, and layout.

### 2. Model Loading

-   The function `load_qa_pipeline(model_path)`:

    -   Loads the tokenizer (`AutoTokenizer`) and model
        (`TFAutoModelForQuestionAnswering`) from the given directory.\
    -   Creates a Hugging Face `pipeline("question-answering")` object.\
    -   Uses Streamlit's `@st.cache_resource` to prevent redundant
        reloading.

-   Model paths are defined for BERT, ALBERT, and RoBERTa.

-   All models are loaded at startup, with error handling if paths are
    invalid.

### 3. User Interface

-   **Model Selection:** Dropdown menu lets users pick BERT, ALBERT, or
    RoBERTa.\
-   **Context Input:** Large text area for pasting or typing the
    passage.\
-   **Question Input:** Single-line text field for asking a question.

### 4. Inference

-   On clicking "Find the Answer":
    -   The chosen pipeline processes the question and context.\
    -   The answer and confidence score are displayed.
-   Error handling ensures user-friendly messages in case of issues.

### 5. Example

Default context:\
\> "The James Webb Space Telescope is the largest optical telescope in
space. Its high resolution and sensitivity allow it to view objects too
old, distant, or faint for the Hubble Space Telescope. It was launched
by an Ariane 5 rocket from Kourou, French Guiana, in 2021."

Default question:\
\> "When was the James Webb telescope launched?"

Expected answer:\
\> "2021" (with confidence score).

------------------------------------------------------------------------

## Conclusion

This project demonstrates the **full lifecycle of building a QA
system**: 
1. Preprocessing and fine-tuning pretrained transformer models
(BERT, ALBERT, RoBERTa) on the SQuAD dataset.\
2. Saving models for reuse.\
3. Deploying them in an interactive Streamlit app for real-time QA
tasks.

The separation of training (Jupyter Notebook) and deployment (Streamlit
App) ensures modularity, reproducibility, and user accessibility.

------------------------------------------------------------------------
