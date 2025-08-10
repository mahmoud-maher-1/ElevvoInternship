# AG News Classification

## 1. Introduction
This repository presents a machine learning pipeline for text classification on the **AG News** dataset. The objective is to build and evaluate models capable of categorizing news articles into four predefined classes:  
1. World  
2. Sports  
3. Business  
4. Science/Technology  

The project follows a structured approach, involving data preprocessing, feature extraction, model training, and evaluation.

---

## 2. Dataset
**Source:** [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
The dataset consists of labeled news headlines and descriptions, with the following attributes:
- **Class Index**: Integer values (1–4) representing the category.  
- **Title**: News headline.  
- **Description**: News article summary.  

A total of **120,000 training samples** are used, with balanced representation across all categories.

---

## 3. Data Preprocessing
To enhance model performance, the raw text undergoes multiple preprocessing steps:
1. **Lowercasing** – Standardizes all tokens to lowercase.
2. **Punctuation and Digit Removal** – Eliminates non-alphabetic characters using regex.
3. **Tokenization** – Splits text into individual tokens using `nltk.word_tokenize`.
4. **Stopword Removal** – Filters out common English stopwords.
5. **Lemmatization** – Converts words to their base forms via `WordNetLemmatizer`.

---

## 4. Exploratory Data Analysis (EDA)
Before modeling, data distribution and word usage patterns are examined:
- **Category Distribution** – Verified class balance through normalized counts.
- **Word Clouds** – Generated for each category to visualize most frequent terms.
- **Term Frequency Analysis** – Identified high-frequency tokens per class using `collections.Counter`.

---

## 5. Feature Extraction
The processed text is transformed into numerical feature vectors using:
- **TF-IDF Vectorization** (`sklearn.feature_extraction.text.TfidfVectorizer`)
  - Captures term importance relative to the dataset.
  - Configured for unigrams, bigrams and trigrams and default weighting.

---

## 6. Model Training
Three supervised learning models are implemented:

1. **Logistic Regression**
   - Multinomial classification setting.
   - Regularization and solver set to defaults for baseline performance.

2. **Random Forest Classifier**
   - Ensemble of decision trees to capture non-linear relationships.
   - Used `300` as the number of estimators.

3. **XGBoost Classifier**
   - Gradient boosting approach optimized for classification tasks.
   - Configured with some default parameters for initial testing.

4. **Feed Forward Neural Network**
   - For better understanding and further testing.
---

## 7. Model Evaluation
Models are evaluated using:
- **Classification Report** – Precision, Recall, F1-score per class.
- **Confusion Matrix** – Visual representation of misclassifications.
- **Macro and Weighted Averages** – For balanced performance measurement.

The confusion matrix is visualized using `sklearn.metrics.ConfusionMatrixDisplay` for intuitive error analysis.

---

## 8. Results Summary
- Logistic Regression achieved the highest macro-average F1-score among tested models.
- Random Forest and XGBoost performed competitively, but with marginally lower recall in certain categories.
- TF-IDF proved effective in capturing discriminative terms for classification.
