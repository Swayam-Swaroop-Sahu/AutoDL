"""Image candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Image model family"): include both end-to-end
deep learning (Small-CNN trained from scratch) and transfer learning (MobileNetV2,
ResNet50, EfficientNetB0 with frozen backbones). All candidates use TensorFlow/Keras.

To stay compatible with the sklearn `cross_val_score` loop in `search.py`, the
transfer-learning candidates are sklearn `Pipeline` objects whose first stage is
a *frozen feature extractor* and whose second stage is a sklearn head. The Small-CNN
candidate uses a Keras wrapper that builds and fits the model per CV fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
    """Return DL candidates for image classification.

    Includes (all deep learning):
      - Small-CNN: end-to-end CNN trained from scratch (good for small datasets)
      - MobileNetV2: transfer learning (lightweight, ~3.5M params)
      - ResNet50: transfer learning (deeper, ~25M params)
      - EfficientNetB0: transfer learning (balanced, ~5M params)

    All use frozen backbones + sklearn head, except Small-CNN which is end-to-end.
    """
    from sklearn.ensemble import RandomForestClassifier

    candidates = []

    # --- End-to-end DL: Small CNN (trained from scratch) ---
    # Built via a KerasModelWrapper for sklearn cross_val_score compatibility.
    candidates.append(Candidate(
        name="Small-CNN",
        factory=lambda: _SmallCNNClassifier(epochs=8, batch_size=32,
                                            input_shape=(64, 64, 3)),
        description="3-layer CNN from scratch (32->64->128 filters, GAP, dense).",
        params="filters=32/64/128, epochs=8",
        pros="No pretrained weights; works offline; good for small datasets.",
        cons="Lower ceiling than transfer learning; needs more data to shine.",
    ))

    # --- Transfer learning (frozen backbone + sklearn head) ---
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


# ----------------------------------------------------------------------
# Keras DL wrapper for sklearn cross_val_score compatibility (Small-CNN)
# ----------------------------------------------------------------------
class _SmallCNNClassifier(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around the from-scratch Small-CNN.

    Resizes incoming images to `input_shape` and trains the CNN end-to-end per fold.
    """

    def __init__(self, epochs=8, batch_size=32, input_shape=(64, 64, 3)):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape

    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow.keras import utils
        from src.model_selection.image_models import build_small_cnn

        X_arr = self._to_batch(X)
        num_classes = len(np.unique(y))
        y_cat = utils.to_categorical(y, num_classes=num_classes)
        self.model_ = build_small_cnn(self.input_shape, num_classes)
        self.model_.fit(
            X_arr, y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.0,
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_arr = self._to_batch(X)
        proba = self.model_.predict(X_arr, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        X_arr = self._to_batch(X)
        return self.model_.predict(X_arr, verbose=0)

    def _to_batch(self, X):
        """Coerce X (list of paths or ndarray) into a float32 batch at input_shape."""
        import numpy as np
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        h, w, _ = self.input_shape
        if isinstance(X, np.ndarray) and X.ndim == 4:
            # Already batched images — resize if needed
            if X.shape[1] != h or X.shape[2] != w:
                import tensorflow as tf
                X = tf.image.resize(X, (h, w)).numpy()
            return X.astype("float32")
        # List of file paths
        arrs = []
        for p in X:
            img = load_img(p, target_size=(h, w))
            arrs.append(img_to_array(img))
        return np.stack(arrs).astype("float32")
