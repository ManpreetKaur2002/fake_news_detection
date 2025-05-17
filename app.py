from flask import Flask, request, render_template, Response
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import joblib

# Initialize Flask
app = Flask(__name__)

# === Load Models ===
q_table = np.load('model/q_table.npy')  # Trained Q-table
vectorizer = joblib.load('model/vectorizer.joblib')
label_encoder = joblib.load('model/label_encoder.joblib')
# vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))  # TF-IDF Vectorizer
# label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))  # LabelEncoder

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).lower()

def state_index(vec):
    return int(np.sum(vec) * 1000) % 5000

def predict_fake_news(text):
    cleaned = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned]).toarray()[0]
    state_idx = state_index(tfidf_vector)
    action = np.argmax(q_table[state_idx])
    label = label_encoder.inverse_transform([action])[0]
    print(label)
    return label

# === Web Routes ===
@app.route('/')
def home():
    # return "Hello from Flask on Vercel!"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    result = predict_fake_news(news_text)  
    print(result)
    return render_template('index.html', news=news_text, prediction=result, message=None)
    
@app.route('/feedback', methods=['POST'])
def feedback():
    news_text = request.form['news']
    prediction = request.form['prediction']
    feedback = request.form['feedback']

    cleaned = clean_text(news_text)
    tfidf_vector = vectorizer.transform([cleaned]).toarray()[0]
    state_idx = state_index(tfidf_vector)
    action = label_encoder.transform([prediction])[0]

    # Reward logic
    reward = 1 if feedback == 'correct' else -1
    learning_rate = 0.1
    q_table[state_idx][action] = q_table[state_idx][action] + learning_rate * (reward - q_table[state_idx][action])

    # Save updated Q-table
    # np.save('model/q_table.npy', q_table)

    return render_template('index.html', prediction=None, news=None, message="✅ Thank you for your feedback!")


# def handler(environ, start_response):
#     return app(environ, start_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))