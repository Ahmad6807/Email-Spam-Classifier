import pickle
import os
import numpy as np
from preprocessing import transform_text, extract_metadata

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_artifacts():
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_name.txt'), 'r') as f:
        model_name = f.read().strip()
    return model, vectorizer, le, model_name


def predict_message(text):
    model, vectorizer, le, model_name = load_artifacts()

    processed = transform_text(text)
    tfidf = vectorizer.transform([processed])
    meta = extract_metadata(text)
    meta_arr = np.array([[meta['char_count'], meta['word_count'], meta['sentence_count']]])
    features = np.hstack((tfidf.toarray(), meta_arr))

    proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
    prediction = model.predict(features)[0]
    label = le.inverse_transform([prediction])[0]
    confidence = max(proba) if proba is not None else None

    return {
        'label': label,
        'confidence': float(confidence) if confidence is not None else None,
        'model': model_name,
    }
