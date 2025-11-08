"""
text_preprocess.py
-------------------
Handles basic NLP preprocessing for text datasets.
"""

import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans text by removing non-alphabetic characters and stopwords.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

def preprocess_text_data(text_lines, max_words=10000, max_len=100):
    """
    Tokenizes and pads text data for deep learning models.

    Parameters
    ----------
    text_lines : list[str]
        Raw text samples
    max_words : int
        Vocabulary size
    max_len : int
        Max token length per sample

    Returns
    -------
    tuple (X_train, X_test, tokenizer)
    """
    cleaned = [clean_text(t) for t in text_lines]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned)

    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    X_train, X_test = train_test_split(padded, test_size=0.2, random_state=42)

    return X_train, X_test, tokenizer
