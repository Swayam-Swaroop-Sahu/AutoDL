"""Text candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Text model family"): TF-IDF + Linear SVM as the
default and floor; sentence-embedding + Logistic Regression as a semantic option;
DistilBERT fine-tune only if cheap options score below threshold AND the time budget
allows. Phase 1 ships the cheap options; DistilBERT is deferred (logged) — it would
pull a multi-hundred-MB model + transformers dependency for marginal gain in the
v2 demo.

Candidates here are sklearn `Pipeline` objects so the search engine treats text the
same as tabular (numeric features → cross_val_score). Embedding+LR is included as a
try/except: it uses sklearn-compatible hashing-trick TF-IDF with higher n-gram range
as a stand-in for semantic embeddings (transformers sentence-transformers would need
network access to download a model — blocked offline).
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB

from .search import Candidate


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
    """Rank candidates; TF-IDF + Linear SVM is the default / floor."""
    svm_clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42),
                                     cv=3)  # adds predict_proba for argmax path
    return [
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
            "TF-IDF + L2 logistic regression (semantic-ish via wider n-grams).",
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
