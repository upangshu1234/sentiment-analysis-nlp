# Sentiment Analysis using NLP 📝

## 📌 Project Overview

This project performs **sentiment analysis** on tweets and user text data using **Natural Language Processing (NLP)** and **Machine Learning**. The dataset contains tweets along with metadata such as time, age group, and country.

The pipeline includes:

* **Data Preprocessing** → Cleaning, tokenization, normalization, stopword removal, stemming, lemmatization
* **Exploratory Data Analysis (EDA)** → Class distribution, word frequency analysis, histograms
* **Feature Engineering** → TF-IDF vectorization
* **Model Training** → Logistic Regression, Decision Tree, Random Forest
* **Evaluation** → Accuracy, Precision, Recall, F1-Score, Confusion Matrix
* **Manual Testing Function** → Predicts sentiment for custom user input

---

## 📂 Dataset

* **Source:** [Kaggle - Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)
* **Columns:**

  * `textID` → Unique identifier of the tweet
  * `text` → Original tweet text
  * `selected_text` → Extracted key text snippet
  * `sentiment` → Sentiment label (negative, neutral, positive)
  * `Time of Tweet`, `Age of User`, `Country`, `Population -2020`, `Land Area (Km²)`, `Density (P/Km²)` → Metadata features

---

## ⚙️ Preprocessing Steps

✔ Remove HTML tags, URLs, punctuation, digits
✔ Tokenization using **NLTK**
✔ Text normalization (lowercasing, trimming spaces)
✔ Stopword removal
✔ Stemming (Lancaster Stemmer) & Lemmatization (WordNet Lemmatizer)
✔ Conversion of categorical features (Time, Country, Age) into numerical codes
✔ TF-IDF vectorization for ML models

---

## 📊 Exploratory Data Analysis

* **Sentiment distribution:**

  * Neutral → 11,117 samples
  * Positive → 8,582 samples
  * Negative → 7,781 samples
* Visualizations:

  * Bar plots of class distribution
  * Histogram of sentiment classes
  * Word frequency distribution

---

## 🤖 Machine Learning Models

Implemented and evaluated:

* **Logistic Regression** → Accuracy **82.9%**
* **Decision Tree Classifier** → Accuracy **75.7%**
* **Random Forest Classifier** → Accuracy **81.1%**

Baseline (majority class prediction) → **40.4%**

**Best Model:** Logistic Regression

---

## 📈 Results

| Model                    | Accuracy | Precision | Recall | F1-score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression      | 82.9%    | ~0.83     | ~0.82  | ~0.83    |
| Decision Tree Classifier | 75.7%    | ~0.76     | ~0.76  | ~0.76    |
| Random Forest Classifier | 81.1%    | ~0.82     | ~0.80  | ~0.81    |

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:**

  * Data: Pandas, NumPy
  * NLP: NLTK, Regex
  * Visualization: Matplotlib, Seaborn
  * ML: Scikit-learn (TF-IDF, Logistic Regression, Decision Tree, Random Forest)

---

## 🚀 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook

   ```bash
   jupyter notebook "Sentiment Analysis by NLP.ipynb"
   ```

4. (Optional) Test custom input

   ```python
   text = "I am very happy today!"
   manual_testing(text)
   ```

---

## ✨ Author

👤 **Upangshu Basak**

