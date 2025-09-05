# Machine Learning Project and NLP Portfolio

## Overview

This portfolio comprises seven comprehensive projects in Natural Language Processing (NLP) and Machine Learning, each addressing distinct challenges in text analysis, classification, summarization, and information extraction. The projects demonstrate both traditional machine learning approaches and modern transformer-based techniques, showcasing a wide range of NLP applications.

## Task Summaries

### 1. IMDB Sentiment Analysis with Logistic Regression & Naive Bayes

This project performs binary sentiment analysis on IMDB movie reviews using classical machine learning models. The implementation includes comprehensive text preprocessing, TF-IDF feature extraction, and evaluation of both Logistic Regression and Naive Bayes classifiers. The system successfully classifies reviews as positive or negative with high accuracy, demonstrating the effectiveness of traditional ML approaches for sentiment analysis tasks.

### 2. Automated News Summarization

This project develops an automated text summarization system that condenses lengthy news articles into concise summaries. The implementation explores both abstractive summarization using the T5 Transformer model and extractive summarization using the LexRank algorithm. The system was evaluated using ROUGE scores, with T5 producing more human-like summaries while LexRank ensured factual accuracy by extracting sentences directly from source text.

### 3. Transformer-Based Question Answering System

This project implements a comprehensive question answering system using three transformer architectures: BERT, ALBERT, and RoBERTa. The models were fine-tuned on the Stanford Question Answering Dataset (SQuAD) and deployed through an interactive Streamlit application. The system demonstrates the full lifecycle of building a QA system, from data preprocessing and model training to deployment and user interaction.

### 4. Unsupervised Topic Modeling on BBC News Dataset

This project applies unsupervised learning techniques to automatically discover latent topics in BBC news articles. The implementation compares two topic modeling approaches: Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF). The system successfully identified coherent topics corresponding to common news categories such as politics, business, sports, technology, and entertainment without using predefined labels.

### 5. Named Entity Recognition (NER) System

This project implements and compares two approaches to Named Entity Recognition: model-based using spaCy's pre-trained transformers and rule-based using custom dictionaries. The system identifies entities such as persons, organizations, and locations within text, with the model-based approach demonstrating higher accuracy and adaptability while the rule-based method provided simplicity and interpretability.

### 6. Fake News Detection System

This project develops a machine learning system to classify news articles as real or fake. The implementation includes comprehensive text preprocessing, exploratory data analysis, feature extraction using TF-IDF, and evaluation of multiple classifiers including Logistic Regression and Support Vector Machines. The system effectively distinguishes between legitimate and fraudulent news content with high accuracy.

### 7. AG News Classification Pipeline

This project implements a multi-class text classification system for categorizing news articles into four categories: World, Sports, Business, and Science/Technology. The pipeline includes extensive text preprocessing, feature extraction using TF-IDF, and evaluation of multiple models including Logistic Regression, Random Forest, XGBoost, and Neural Networks. The system demonstrates effective categorization of news content with strong performance metrics.

## Technical Tools and Libraries

The following tools and libraries were used across the projects:

### Natural Language Processing
- NLTK (Natural Language Toolkit) - Text preprocessing and tokenization
- spaCy - Advanced NLP processing and named entity recognition
- Transformers (Hugging Face) - Transformer models (BERT, ALBERT, RoBERTa, T5)
- WordCloud - Text visualization
- sumy - Text summarization utilities

### Machine Learning Frameworks
- Scikit-learn - Traditional machine learning algorithms and utilities
- TensorFlow - Deep learning framework
- PyTorch - Deep learning framework
- XGBoost - Gradient boosting framework

### Data Handling and Processing
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- SciPy - Scientific computing

### Visualization
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- pyLDAvis - Interactive topic model visualization

### Deployment and Utilities
- Streamlit - Web application deployment
- Joblib - Model serialization
- datasets - Dataset handling utilities
- rouge_score - Text summarization evaluation
- seqeval - Sequence labeling evaluation

### Specialized Libraries
- TF-IDF Vectorization - Feature extraction from text
- LexRank - Extractive summarization algorithm
- LDA/NMF - Topic modeling algorithms

## Conclusion

This portfolio demonstrates comprehensive expertise in natural language processing and machine learning, covering a wide spectrum of techniques from traditional machine learning approaches to modern transformer-based architectures. The projects showcase practical applications in sentiment analysis, text summarization, question answering, topic modeling, named entity recognition, fake news detection, and text classification. The diverse toolkit of technologies employed across these projects highlights proficiency in both theoretical concepts and practical implementation of NLP systems.
