# Unsupervised NLP on BBC News Dataset

## Overview
This project applies **unsupervised Natural Language Processing (NLP)** techniques to analyze and categorize **BBC news articles**. Unlike supervised approaches, no predefined labels were used. Instead, the system automatically identified **hidden topics** from the dataset using **topic modeling** methods.

The project compares two widely used unsupervised algorithms:
1. **Latent Dirichlet Allocation (LDA)**  
2. **Non-Negative Matrix Factorization (NMF)**  

Both approaches provide insights into the dominant themes present in the news collection.

---

## Objectives
- Preprocess and clean the text data to make it suitable for analysis.  
- Extract features from text using **Bag-of-Words (BoW)** and **TF-IDF** representations.  
- Apply **unsupervised topic modeling** to discover dominant themes.  
- Compare the interpretability and performance of **LDA** and **NMF**.  
- Visualize the results to better understand topic structures within the dataset.  

---

## Methodology

### 1. Data Preparation
- Dataset: **BBC News dataset** (CSV format).  
- Removed irrelevant attributes (IDs, links, publication dates).  
- Preprocessing steps included:  
  - Lowercasing text  
  - Removing punctuation  
  - Normalizing and preparing sentences  

### 2. Feature Extraction
- **Bag-of-Words (BoW):** Captured word frequencies across documents.  
- **TF-IDF (Term Frequency – Inverse Document Frequency):** Weighted terms by importance, reducing the influence of very common words.  

### 3. Topic Modeling
- **Latent Dirichlet Allocation (LDA):**  
  - A probabilistic model assuming each document is a mixture of topics.  
  - Provided probability distributions of words per topic and topics per document.  

- **Non-Negative Matrix Factorization (NMF):**  
  - A mathematical decomposition technique.  
  - Produced clearer and more distinct sets of topics compared to LDA.  

- Both models were configured to extract **7 topics** from the dataset.  

### 4. Visualization
- **pyLDAvis** was used for interactive visualization of LDA results.  
- Top keywords per topic were extracted for both LDA and NMF, allowing clear interpretation of themes such as:  
  - **Politics**  
  - **Business**  
  - **Sports**  
  - **Technology**  
  - **Entertainment**  

---

## Key Findings
- **LDA**: Useful for showing overlaps between topics, capturing probabilistic relationships across articles.  
- **NMF**: Produced sharper, more distinct clusters, making interpretation of topics easier.  
- Both methods uncovered themes that align with common BBC news categories.  

---

## Conclusion
This project demonstrates how **unsupervised NLP techniques** can automatically uncover meaningful patterns in large text datasets. By applying **topic modeling**, the system successfully extracted dominant themes in BBC news articles without requiring labeled data.

### Applications
- **Media monitoring** – automatically tracking emerging topics in news feeds.  
- **Business intelligence** – summarizing large volumes of reports and documents.  
- **Content recommendation** – suggesting articles based on shared topics.  
- **Research analysis** – organizing large corpora of academic or policy documents.  

---

## Technologies Used
- **Python**  
- **NLTK** – text preprocessing  
- **scikit-learn** – feature extraction (BoW, TF-IDF) and topic modeling (LDA, NMF)  
- **pyLDAvis** – interactive topic visualization  
- **Pandas** – dataset handling  

---

## Dataset
- **BBC News Dataset** – a collection of news articles from BBC, covering multiple categories including business, politics, sports, technology, and entertainment.  

---
