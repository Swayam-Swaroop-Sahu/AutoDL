# explainDL/preprocessing/text_preprocessor.py

import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


class TextPreprocessor:
    """
    Handles:
    - Text cleaning
    - Tokenization
    - Sequence generation
    - Label encoding
    """

    def __init__(self, max_words=10000, max_len=120):
        self.max_words = max_words
        self.max_len = max_len

        self.tokenizer = None
        self.label_encoder = None

    # ------------------------------------------------------------------
    def clean(self, text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        return " ".join(
            [word for word in text.split() if word not in STOPWORDS]
        )

    # ------------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------------
    def fit_transform(self, texts, labels):
        cleaned = [self.clean(t) for t in texts]

        # Tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<UNK>")
        self.tokenizer.fit_on_texts(cleaned)

        seq = self.tokenizer.texts_to_sequences(cleaned)
        X = pad_sequences(seq, maxlen=self.max_len)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        return X, y

    # ------------------------------------------------------------------
    # PREDICTION MODE
    # ------------------------------------------------------------------
    def transform(self, texts):
        cleaned = [self.clean(t) for t in texts]
        seq = self.tokenizer.texts_to_sequences(cleaned)
        X = pad_sequences(seq, maxlen=self.max_len)
        return X
