# Automated News Summarization

## Overview  
This project focuses on developing an **automated text summarization system** that condenses lengthy news articles into short, meaningful summaries.  
The dataset used is the **CNN/Daily Mail news dataset**, which contains thousands of articles paired with human-written summaries.  

The system is built using **Natural Language Processing (NLP)** techniques and a **Transformer-based model** called T5, alongside **LexRank**, an extractive summarization method.  
The goal is to help readers quickly understand the main points of news articles without reading them in full.  

---

## Project Steps  

### 1. Data Preparation  
- The news dataset was divided into **training, validation, and test sets**.  
- A limited portion of the dataset was used to fit available computing resources.  

### 2. Preprocessing  
- Articles were cleaned by removing unnecessary symbols and formatting.  
- Stop words (e.g., “the,” “is,” “at”) were carefully managed to preserve meaning.  

### 3. Model Development  
The project explored **two summarization approaches**:  

#### a) Abstractive Summarization with T5  
- An **abstractive summarization model** was developed using the **T5 Transformer**.  
- The model was trained to generate new sentences that capture the essence of the original text, rather than copying existing ones.  

#### b) Extractive Summarization with LexRank  
- **LexRank**, a graph-based algorithm, was also applied to extractive summarization.  
- Instead of generating new text, LexRank selects the **most important sentences** directly from the article.  
- This ensures summaries remain true to the original wording, though less flexible than T5.  

### 4. Evaluation  
- Performance was measured using **ROUGE scores**, comparing generated summaries with human-written ones.  
- T5 provided more **human-like summaries**, while LexRank produced **factual extracts**.  
- The combination showed how extractive and abstractive methods differ in style and application.  

---

## Key Outcomes  
- A working **automated summarization tool** was created using both extractive and abstractive methods.  
- **T5** demonstrated the ability to create fluid, natural-sounding summaries.  
- **LexRank** ensured factual accuracy by directly pulling sentences from the source text.  
- Results confirmed the feasibility of combining different summarization strategies in resource-limited environments.  

---

## Practical Applications  
This summarization system can be applied to:  
- **News platforms** – Delivering shorter versions of articles to improve reader engagement.  
- **Business intelligence** – Quickly digesting large volumes of reports.  
- **Education and research** – Helping students and analysts focus on key insights without reading entire documents.  

---
