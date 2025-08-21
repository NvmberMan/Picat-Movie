# app.py

import joblib
import warnings
import re
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. Load Pre-trained Components ---
try:
    lr_model = joblib.load('model/lr_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    mlb = joblib.load('model/multilabel_binarizer.pkl')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("Model, vectorizer, and binarizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    lr_model = None
    vectorizer = None
    mlb = None

# --- 2. Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    tokens = word_tokenize(text.lower())
    tokens_no_stopwords = [word for word in tokens if word not in stop_words]
    tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
    cleaned_tokens = [
        token for token in tokens_lemmatized
        if re.fullmatch(r'[a-z]+', token) and token != ''
    ]
    return ' '.join(cleaned_tokens)

# --- 3. Define a Prediction Endpoint with a Custom Threshold ---
@app.route('/predict', methods=['POST'])
def predict():
    if not lr_model or not vectorizer or not mlb:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data = request.get_json()
        title = data.get('title', '')
        description = data.get('description', '')
        
        # Ambil nilai threshold dari permintaan JSON, atau gunakan nilai default 0.5
        threshold_str = data.get('threshold', '0.5')
        
        try:
            threshold = float(threshold_str)
        except ValueError:
            return jsonify({"error": "Threshold must be a valid number."}), 400

        if not title and not description:
            return jsonify({"error": "Either 'title' or 'description' (or both) must be provided."}), 400

        # Combine title and description
        combined_text = f"{title} {description}".strip()
        
        # Preprocess the combined text
        processed_text = preprocess_text(combined_text)

        # Transform the text using the loaded vectorizer
        X_new = vectorizer.transform([processed_text])
        
        # Get probability predictions for each genre
        y_pred_proba = np.array([
            estimator.predict_proba(X_new)[:, 1] for estimator in lr_model.estimators_
        ]).T

        # Apply the user-defined threshold to the probabilities
        y_pred_binary = (y_pred_proba > threshold).astype(int)

        # Get the predicted genres based on the binary predictions
        predicted_genres = mlb.inverse_transform(y_pred_binary)

        # Handle cases where no genres are predicted
        if not predicted_genres or not predicted_genres[0]:
            # If no genres pass the threshold, predict the genre with the highest probability
            top_genre_index = np.argmax(y_pred_proba)
            top_genre = mlb.classes_[top_genre_index]
            predicted_genres = [(top_genre,)]
            
        genre_list = predicted_genres[0]
        
        return jsonify({
            "title": title,
            "description": description,
            "predicted_genres": genre_list,
            "threshold_used": threshold
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True)