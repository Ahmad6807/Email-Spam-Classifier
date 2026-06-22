import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from preprocessing import preprocess_dataframe

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_and_clean_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    df = df[['label', 'message']].dropna().drop_duplicates()
    return df


def train_and_evaluate():
    df = load_and_clean_data(DATA_PATH)

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['label'])

    df = preprocess_dataframe(df, text_column='message')

    feature_cols = ['char_count', 'word_count', 'sentence_count']
    X_meta = df[feature_cols].values
    y = df['target'].values

    X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        df['processed_text'], X_meta, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_meta))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_meta))

    models = {
        'MultinomialNB': MultinomialNB(),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=50, random_state=42),
    }

    results = {}
    best_model = None
    best_precision = 0
    best_name = ''

    for name, model in models.items():
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'precision': prec}
        print(f"{name:20s}  Accuracy: {acc:.4f}  Precision: {prec:.4f}")

        if prec > best_precision:
            best_precision = prec
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with Precision: {best_precision:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    with open(os.path.join(MODEL_DIR, 'model_name.txt'), 'w') as f:
        f.write(best_name)

    print("Models saved to models/ directory")

    return results, best_name


if __name__ == '__main__':
    train_and_evaluate()
