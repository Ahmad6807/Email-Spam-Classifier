import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_training_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam.csv')
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    df = df[['label', 'message']].dropna().drop_duplicates()
    return df


def get_spam_keywords(top_n=20):
    df = load_training_data()
    spam_texts = df[df['label'] == 'spam']['message']
    all_words = ' '.join(spam_texts.astype(str)).lower().split()
    stop_words = set([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could',
        'shall', 'should', 'may', 'might', 'must', 'to', 'of', 'in', 'for', 'on',
        'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'because', 'as', 'until', 'while',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
        'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'whose', 'about', 'if', 'but', 'and', 'or',
        'up', 'down', 'like', 'also', 'get', 'got',
    ])
    words = [w for w in all_words if w.isalpha() and len(w) > 2 and w not in stop_words]
    return Counter(words).most_common(top_n)


def get_spam_ham_distribution():
    df = load_training_data()
    return df['label'].value_counts().to_dict()
