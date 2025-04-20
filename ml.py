import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load dataset
df = pd.read_csv("train.csv")

# Check dataset columns
print(df.columns)

# Drop missing values
df.dropna(inplace=True)
print("done drop na")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply cleaning to text column
df['text'] = df['text'].apply(clean_text)
print("done cleaning")

# Extract features and labels
X = df['text']
y = df['label']

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
print("done vectorizing")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
print("done splitting")

# Model Initialization
log_reg = LogisticRegression()
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
log_reg.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print("trained")

# Evaluate models
def evaluate_model(model, X_test, y_test):
    print("evaluating models")
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

# Store results
results = {
    "Logistic Regression": evaluate_model(log_reg, X_test, y_test),
    "SVM": evaluate_model(svm_model, X_test, y_test),
    "Random Forest": evaluate_model(rf_model, X_test, y_test)
}

# Print results
for model, scores in results.items():
    print(f"\n{model} Performance:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

# Plot confusion matrix for Random Forest
plt.figure(figsize=(5, 5))
cm = confusion_matrix(y_test, rf_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()
