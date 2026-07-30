# src/model_selection/tabular_models.py
"""
Candidate tabular neural architectures (Phase 1c — unified multiclass).

All MLPs use a single code path:
  - final Dense(units=num_classes, activation="softmax")
  - loss="sparse_categorical_crossentropy"
  - Integer labels (via LabelEncoder)
  - Decoded via argmax(predict_proba, axis=1)

No binary-vs-multiclass branches remain — predict_proba is always
shape (N, num_classes) and we always argmax to decode.
"""

from typing import Union
from tensorflow.keras import models, layers, optimizers


def _finalize(model: models.Model, num_classes: int):
    """Add a unified softmax head and compile with sparse_categorical_crossentropy."""
    # BUGFIX Phase 1c: removed `if num_classes == 2` branch.
    # Always softmax + sparse_categorical_crossentropy.
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp_small(num_features: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Input(shape=(num_features,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    return _finalize(model, num_classes)


def build_mlp_medium(num_features: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Input(shape=(num_features,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    return _finalize(model, num_classes)


def build_mlp_large(num_features: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Input(shape=(num_features,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    return _finalize(model, num_classes)
