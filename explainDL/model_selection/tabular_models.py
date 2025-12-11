# explainDL/model_selection/tabular_models.py
"""
Candidate tabular neural architectures.
These are lightweight MLPs intended for quick evaluation and training.
They are compiled with a loss compatible with integer labels (sparse categorical)
when multiclass, and binary_crossentropy for binary classification.
"""

from typing import Union
from tensorflow.keras import models, layers, optimizers


def _finalize(model: models.Model, num_classes: int):
    """
    Adds the final classification head (if not already added) and compiles.
    We use:
      - binary_crossentropy + sigmoid for binary (num_classes == 2)
      - sparse_categorical_crossentropy + softmax for multiclass
    The trainer passes integer labels, so sparse loss is appropriate.
    """
    if num_classes == 2:
        # Ensure single logit output for binary classification
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
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
