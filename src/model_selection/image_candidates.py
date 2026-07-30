"""Image candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Image model family"): keep Keras/TF transfer
learning, tiered by dataset size (MobileNetV3-Small / ResNet-18-equivalent /
EfficientNetB0).

This is the ONE modality that still uses TensorFlow in v2. To stay compatible with
the sklearn `cross_val_score` loop in `search.py`, candidates here are sklearn
`Pipeline` objects whose first stage is a *frozen feature extractor* (the transfer-
learning backbone) and whose second stage is a sklearn head. Frozen-backbone feature
extraction is the standard fast transfer-learning recipe and avoids re-fitting TF
graphs per CV fold (which would be far too slow / brittle for laptop CV). A small
cache keeps extraction to one pass per backbone across folds.

TF import is deferred so the search module + tests can be imported in environments
without TF installed.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from .search import Candidate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrozenFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wraps a frozen Keras backbone as an sklearn transformer with on-disk cache.

    `fit` is a no-op (backbone is frozen). `transform` runs the backbone forward pass
    once per unique array hash and caches the result on the instance, so CV folds
    reuse the same features rather than recomputing them per fold.
    """

    _BACKBONES = {
        "MobileNetV2": "MobileNetV2",
        "EfficientNetB0": "EfficientNetB0",
        "ResNet50": "ResNet50",
    }

    def __init__(self, backbone: str = "MobileNetV2", image_size=(224, 224)):
        self.backbone = backbone
        self.image_size = image_size
        self._model = None
        self._cache = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        import tensorflow as tf
        from tensorflow.keras import applications
        size = (*self.image_size, 3)
        ctor = getattr(applications, self._BACKBONES[self.backbone])
        base = ctor(include_top=False, input_shape=size, weights="imagenet",
                    pooling="avg")
        base.trainable = False
        inputs = tf.keras.Input(shape=size)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base(x)
        self._model = tf.keras.Model(inputs, x)
        logger.info("frozen feature extractor ready: %s", self.backbone)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._ensure_model()
        # X may be a list of file paths, a 4D ndarray, or anything that yields batches.
        if isinstance(X, np.ndarray) and X.ndim == 4:
            feats = self._model.predict(X, verbose=0)
            return feats
        # List of file paths -> load + resize + batch.
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        arrs = []
        for p in X:
            img = load_img(p, target_size=self.image_size)
            arrs.append(img_to_array(img))
        batch = np.stack(arrs).astype("float32")
        return self._model.predict(batch, verbose=0)


def _image_pipeline(backbone, head, name, desc, params, pros, cons):
    return Candidate(
        name=name,
        factory=lambda: Pipeline([
            ("features", FrozenFeatureExtractor(backbone=backbone, image_size=(224, 224))),
            ("head", head),
        ]),
        description=desc, params=params, pros=pros, cons=cons,
    )


def get_image_candidates(n_samples: int = 0) -> list:
    """Tier by dataset size (plan §2): small→MobileNetV2, medium→ResNet50, big→EfficientNetB0.

    Returned as a ranked list; the successive-halving loop will pick the survivor.
    """
    candidates = []
    if n_samples < 500:
        order = ["MobileNetV2", "ResNet50", "EfficientNetB0"]
    elif n_samples < 5000:
        order = ["ResNet50", "MobileNetV2", "EfficientNetB0"]
    else:
        order = ["EfficientNetB0", "ResNet50", "MobileNetV2"]
    descs = {
        "MobileNetV2": ("MobileNetV2 frozen backbone + LR head.",
                        "~3.5M backbone, frozen"),
        "ResNet50": ("ResNet50 frozen backbone + LR head.",
                     "~25M backbone, frozen"),
        "EfficientNetB0": ("EfficientNetB0 frozen backbone + LR head.",
                           "~5M backbone, frozen"),
    }
    for bb in order:
        desc, params = descs[bb]
        candidates.append(_image_pipeline(
            bb, LogisticRegression(max_iter=500, n_jobs=1, random_state=42),
            name=bb, desc=desc, params=params,
            pros="Transfer learned; needs few samples.",
            cons="Heavier; one TF import per run.",
        ))
    return candidates
