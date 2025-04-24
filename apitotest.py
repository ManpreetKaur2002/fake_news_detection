import os
import re
import string
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================
# Step 1: Fetch live news
# =====================
API_KEY = "c27476c0182a448fa9c0dc9a6d9f9d1d"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
PARAMS = {
    'q': 'technology OR politics OR finance OR sports',
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 1,
    'apiKey': API_KEY
}

def generate_id(title, content):
    unique_string = (title or '') + (content or '')
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def fetch_articles():
    response = requests.get(NEWS_ENDPOINT, params=PARAMS)
    data = response.json()

    if response.status_code != 200 or not data.get("articles"):
        print("Error or no data fetched")
        return []

    processed = []
    for article in data["articles"]:
        title = article.get('title', '')
        text = article.get('content') or article.get('description') or ''
        if not text.strip():
            continue
        processed.append({
            'id': generate_id(title, text),
            'title': title,
            'text': text,
            'label': 0  # Default label = 0 (Fake); modify as needed
        })

    return processed

# =====================
# Step 2: Preprocess text
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# =====================
# Step 3: Main Training Pipeline
# =====================
def run_pipeline():
    # Step 1: Load training dataset
    base_df = pd.read_csv("train.csv")
    base_df.dropna(inplace=True)
    base_df['text'] = base_df['text'].apply(clean_text)

    # Step 2: Load live articles
    new_articles = fetch_articles()
    if not new_articles:
        print("No new articles. Exiting.")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    temp_file = f"news_articles_{date_str}.csv"
    live_df = pd.DataFrame(new_articles)
    live_df['text'] = live_df['text'].apply(clean_text)
    live_df.to_csv(temp_file, index=False)

    # Step 3: Combine datasets
    df = pd.concat([base_df, live_df], ignore_index=True)
    print(f"Total training samples: {len(df)}")

    # Step 4: TF-IDF and labels
    X = df['text']
    y = df['label']
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Step 5: Train models
    log_reg = LogisticRegression()
    svm_model = SVC(probability=True)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    log_reg.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    print("✅ Models trained")

    # Step 6: Save model
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer saved")

    # Step 7: Evaluate
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }

    models = {"Logistic Regression": log_reg, "SVM": svm_model, "Random Forest": rf_model}
    for name, model in models.items():
        print(f"\n🔍 {name} Performance:")
        for metric, val in evaluate_model(model, X_test, y_test).items():
            print(f"{metric}: {val:.4f}")

    # Step 8: Confusion matrix (for Random Forest)
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # Step 9: Delete temp file
    os.remove(temp_file)
    print(f"🗑️ Deleted temporary file: {temp_file}")

# =====================
# Run everything
# =====================
if __name__ == "__main__":
    run_pipeline()
