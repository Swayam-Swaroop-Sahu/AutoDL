"""Text candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Text model family"): include both deep
learning (LSTM, BiLSTM, Text-CNN) and classical ML (TF-IDF + LinearSVC / LR /
ComplementNB). DistilBERT fine-tune is deferred (would pull a multi-hundred-MB
model + transformers dependency for marginal gain in offline settings).

Sklearn `Pipeline` candidates work seamlessly with `cross_val_score` in the
search loop. Keras DL candidates are scored via a small wrapper that builds
and fits the model on the train fold, returning balanced_accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.base import BaseEstimator, ClassifierMixin

from .search import Candidate


# ----------------------------------------------------------------------
# Keras DL wrapper for sklearn cross_val_score compatibility
# ----------------------------------------------------------------------
class KerasTextClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a Keras text model factory.

    `model_fn(vocab_size, max_len, num_classes)` returns a compiled Keras model.
    `fit_transform` fits on integer-encoded sequences, `predict` returns class indices.
    """

    def __init__(self, model_fn, epochs=5, batch_size=32, vocab_size=10000, max_len=100):
        self.model_fn = model_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_len = max_len

    def fit(self, X, y):
        from tensorflow.keras import utils
        num_classes = len(np.unique(y))
        y_cat = utils.to_categorical(y, num_classes=num_classes)
        self.model_ = self.model_fn(self.vocab_size, self.max_len, num_classes)
        self.model_.fit(
            X, y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.0,
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        proba = self.model_.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.model_.predict(X, verbose=0)


def _keras_text_candidate(name, model_fn, epochs, vocab_size, max_len, desc, params, pros, cons):
    return Candidate(
        name=name,
        factory=lambda: KerasTextClassifier(
            model_fn=model_fn,
            epochs=epochs,
            vocab_size=vocab_size,
            max_len=max_len,
        ),
        description=desc, params=params, pros=pros, cons=cons,
    )


def _text_pipeline(clf, name, desc, params, pros, cons):
    return Candidate(
        name=name,
        factory=lambda: Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                      sublinear_tf=True, strip_accents="unicode",
                                      min_df=2, lowercase=True)),
            ("clf", clf),
        ]),
        description=desc, params=params, pros=pros, cons=cons,
    )


def get_text_candidates(n_samples: int = 0) -> list:
    """Return DL + classical ML candidates for text classification.

    Includes:
      - Deep Learning: LSTM, BiLSTM, Text-CNN
      - Classical ML: TFIDF_LinearSVM, TFIDF_LogReg, TFIDF_ComplementNB
    """
    # Import Keras builders (deferred so tests can import this module without TF)
    from src.model_selection.text_models import build_lstm, build_bilstm, build_text_cnn

    # Default vocab/length (overridden by preprocessor in training; search uses defaults)
    VOCAB = 10000
    MAX_LEN = 100

    svm_clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42),
                                     cv=3)  # adds predict_proba for argmax path

    return [
        # --- Deep Learning (Keras) ---
        _keras_text_candidate(
            "BiLSTM",
            model_fn=build_bilstm,
            epochs=5,
            vocab_size=VOCAB,
            max_len=MAX_LEN,
            desc="Bidirectional LSTM with 64 units + dense head.",
            params="embed=128, lstm=64, epochs=5",
            pros="Captures long-range context both directions.",
            cons="Slower training; needs more data than TF-IDF.",
        ),
        _keras_text_candidate(
            "LSTM",
            model_fn=build_lstm,
            epochs=5,
            vocab_size=VOCAB,
            max_len=MAX_LEN,
            desc="Single-layer LSTM (64 units) + dense head.",
            params="embed=128, lstm=64, epochs=5",
            pros="Lighter than BiLSTM; good baseline.",
            cons="Forward-only context.",
        ),
        _keras_text_candidate(
            "Text-CNN",
            model_fn=build_text_cnn,
            epochs=5,
            vocab_size=VOCAB,
            max_len=MAX_LEN,
            desc="1D Conv (kernel=5, filters=128) + global max pool + dense head.",
            params="embed=128, conv=128, epochs=5",
            pros="Fast convergence; captures local n-gram patterns.",
            cons="Limited receptive field vs RNNs.",
        ),

        # --- Classical ML (TF-IDF + sklearn) ---
        _text_pipeline(
            svm_clf, "TFIDF_LinearSVM",
            "TF-IDF uni+bi-grams + Linear SVM (calibrated for probabilities).",
            params="tfidf 20k, ngram 1-2",
            pros="Strong default for text classification; fast; deterministic.",
            cons="Lexical only — no semantics.",
        ),
        _text_pipeline(
            LogisticRegression(max_iter=1000, n_jobs=1, random_state=42),
            "TFIDF_LogReg",
            "TF-IDF + L2 logistic regression.",
            params="tfidf 20k, ngram 1-2, L2",
            pros="Good multiclass; proba-native.",
            cons="More regularization-sensitive than SVM on small data.",
        ),
        _text_pipeline(
            ComplementNB(),
            "TFIDF_ComplementNB",
            "TF-IDF + Complement Naive Bayes (imbalanced-friendly floor).",
            params="tfidf 20k, ngram 1-2",
            pros="Cheap floor for small/imbalanced text.",
            cons="Weak baseline on most real datasets.",
        ),
    ]
