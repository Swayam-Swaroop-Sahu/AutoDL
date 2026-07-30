# src/preprocessing/text_preprocessor.py

import re
import unicodedata
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
    - Text cleaning (Unicode-safe: preserves Chinese, Arabic, etc.)
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
        """Unicode-safe text cleaning (BUGFIX Phase 1e item 5).

        Preserves all script alphabets (Latin, Chinese, Arabic, Cyrillic, ...).
        Earlier `re.sub(r"[^a-zA-Z\\s]", "", text)` destroyed every non-Latin
        character which made Chinese/Arabic documents turn into empty strings.
        We now:
          - Normalize Unicode (NFKC) to fold fullwidth forms etc.
          - Drop only true control characters and noise (HTML tags, URLs).
          - Lowercase for ASCII letter range only; preserve casing of others.
        """
        if text is None:
            return ""
        # 1. Unicode normalize (fold fullwidth -> halfwidth, compatibility forms)
        text = unicodedata.normalize("NFKC", str(text))
        # 2. Drop HTML tags and stray URLs
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        # 3. Drop only ASCII control characters (preserve all scripts)
        text = "".join(ch for ch in text if ch == "\t" or ch == "\n" or not unicodedata.category(ch).startswith("C"))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------------
    def fit_transform(self, texts, labels):
        # Validate inputs
        if not texts or len(texts) == 0:
            raise ValueError("No texts provided for preprocessing. Please provide at least one text sample.")

        if not labels or len(labels) == 0:
            raise ValueError("No labels provided for preprocessing. Please provide labels for each text sample.")

        if len(texts) != len(labels):
            raise ValueError(f"Mismatch between texts ({len(texts)}) and labels ({len(labels)}). They must have the same length.")

        if len(texts) < 2:
            raise ValueError(f"Too few text samples ({len(texts)}). Minimum 2 samples required for training.")

        # Check for sufficient unique labels
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Only {len(unique_labels)} unique label(s) found. Minimum 2 different labels required for classification.")

        cleaned = [self.clean(t) for t in texts]

        # Check that cleaning didn't remove all text (allow Unicode-only inputs)
        if all(not t.strip() for t in cleaned):
            raise ValueError("All texts became empty after cleaning. Please check your text data format.")

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
