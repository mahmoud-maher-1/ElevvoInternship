# IMDB Sentiment Analysis with Logistic Regression & Naive Bayes

This project performs **binary sentiment analysis** (positive/negative) on IMDB movie reviews using machine learning models. It includes preprocessing, TF-IDF feature extraction, classification, and evaluation.

---

## Dataset

- **Source**: [Kaggle: IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews/data)
- **Format**: CSV
- **Fields**:
  - `review`: Full text of the review.
  - `sentiment`: `positive` or `negative`.

---

##  Setup and Dependencies

```python
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
```

### Download NLTK Resources

```python
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Preprocessing

1. Lowercase text
2. Remove punctuation and numbers
3. Tokenization
4. Stopword removal
5. Lemmatization

---

## Feature Extraction

TF-IDF vectorizer to convert text reviews into numerical features.

---

## Label Encoding

`positive` → 1, `negative` → 0 using `LabelEncoder`.

---

## Models Used

### 1. Logistic Regression

- Tuned using `GridSearchCV` with:
```python
{
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

### 2. Multinomial Naive Bayes

- Tuned using:
```python
{
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}
```

---

## Evaluation

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Most frequent words in positive/negative reviews using `Counter`

---

## Visualization

- Confusion matrix plots
- Bar charts of most common words per sentiment class

---

## Results

Both models performed well after tuning. TF-IDF provided informative features. Logistic Regression and Naive Bayes had complementary strengths.

---

## Saving Models for deployment

```python
dump(grid_search_logreg, 'Saved Models/GridSearchforLogisticRegression.pkl')
dump(grid_search, 'Saved Models/GridSearchforMultinomialNaiveBayes.pkl')
dump(tfidf, 'Saved Models/ReviewTfidf.pkl')
dump(le, 'Saved Models/SentimentLabelEncoder.pkl')
```

---

## Conclusion

A complete end-to-end sentiment analysis project demonstrating classic ML-based NLP.

---

## Structure

```
.
├── Dataset/
│   └── IMDB Dataset.csv
├── sentiment_analysis.ipynb
└── README.md
```