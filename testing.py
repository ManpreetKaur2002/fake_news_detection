import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------ Text Preprocessing ------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def preprocess_data(df):
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)
    return df

# ------------------ Feature Extraction ------------------

def extract_features(df, vectorizer=None):
    X = df['text']
    y = df['label']
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
    else:
        X_tfidf = vectorizer.transform(X)

    return X_tfidf, y, vectorizer

# ------------------ Evaluation ------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

# ------------------ Initialization ------------------

vectorizer = TfidfVectorizer(max_features=5000)
label_encoder = LabelEncoder()
incremental_model = SGDClassifier(loss='log_loss', random_state=42)

is_first_batch = True
classes = None

# ------------------ Incremental Training ------------------

def train_incrementally(new_data_path):
    global is_first_batch, vectorizer, incremental_model, classes

    df = pd.read_csv(new_data_path)
    df = preprocess_data(df)

    df['label'] = label_encoder.fit_transform(df['label'])

    X_batch, y_batch, vectorizer = extract_features(df, vectorizer if not is_first_batch else None)

    if is_first_batch:
        classes = np.unique(y_batch)
        incremental_model.partial_fit(X_batch, y_batch, classes=classes)
        is_first_batch = False
    else:
        incremental_model.partial_fit(X_batch, y_batch)

    print(f"Incremental training done on {len(df)} samples.")

# ------------------ Evaluate on Test Data ------------------

def evaluate_on_test(test_data_path):
    df_test = pd.read_csv(test_data_path)
    df_test = preprocess_data(df_test)
    df_test['label'] = label_encoder.transform(df_test['label'])
    X_test, y_test, _ = extract_features(df_test, vectorizer)

    # Additional Models (trained on same data)
    models = {
        "Incremental SGD (LogReg)": incremental_model,
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (LinearSVC)": LinearSVC()
    }

    # Train traditional models (not SGD)
    X_train, _, y_train, _ = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
    for name, model in models.items():
        if name != "Incremental SGD (LogReg)":
            model.fit(X_train, y_train)

    # Evaluate all models
    for name, model in models.items():
        print(f"\n{name} Results:")
        results = evaluate_model(model, X_test, y_test)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    # Confusion Matrix for the Incremental Model
    cm = confusion_matrix(y_test, incremental_model.predict(X_test))
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: Incremental SGD")
    plt.show()

# ------------------ Run Everything ------------------

train_incrementally("train.csv")
evaluate_on_test("train.csv")
