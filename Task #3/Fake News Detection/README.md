Okay, I will generate documentation for the `Fake News Detection.ipynb` notebook, styled similarly to the provided `README.md`.

---

# Fake News Detection using Machine Learning

This project demonstrates how to build a machine learning model to classify news articles as either **REAL** or **FAKE**.

## Dataset Source

The dataset used is the "Fake News Detection" dataset from Kaggle:
[https://www.kaggle.com/datasets/jordanknox/fake-news-detection](https://www.kaggle.com/datasets/jordanknox/fake-news-detection)

## Methodology

The notebook follows these key steps:

1.  **Data Loading:** Reads the `True.csv` and `Fake.csv` datasets.
    </br><br>
2.  **Data Preprocessing:**
    *   Adds a `label` column (`1` for REAL, `0` for FAKE).
    *   Combines the two datasets into one.
    *   Shuffles the combined dataset.
    </br><br>
3.  **Feature Engineering & Text Preprocessing:**
    *   Keeps `title` and `text` columns for proper usage
    *   Cleans the text data by removing punctuation and digits.
    *   Converts text to lowercase.
    </br><br>
4.  **Exploratory Data Analysis (EDA):**
    *   Displays the first few rows of the processed dataset.
    *   Analyzes the distribution of REAL and FAKE news labels.
    *   Visualizes the top words associated with REAL and FAKE news using word clouds.
    </br><br>
5.  **Model Preparation:**
    *   Splits the data into training and testing sets.
    *   Uses `TfidfVectorizer` to convert text content into numerical TF-IDF features.
    </br><br>
6.  **Feature Extraction**
    * Performs Feature extraction on the `title` and `text` columns using two different TFIDF vectorizers with slightly different parameters
    </br><br>
7.  **Model Training & Evaluation:**
    *   Trains a `Logistic Regression` and `Support Vector Classifier` models on the TF-IDF features.
    *   Evaluates the models' performance on the train & test set (to detect overfitting and underfitting) using:
        *   Accuracy Score
        *   Classification Report (Precision, Recall, F1-Score)
        *   Confusion Matrix
    </br><br>
8.  **Results Interpretation:**
    *   Presents the calculated accuracy.
    *   Displays the detailed classification report and confusion matrix to understand the model's effectiveness in distinguishing between REAL and FAKE news.

## Used Tools

*   Python 3.x
*   Jupyter Notebook
*   Pandas
*   Scikit-learn (sklearn)
*   NLTK (Natural Language Toolkit)
*   Matplotlib
*   Seaborn
*   WordCloud

---