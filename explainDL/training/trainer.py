# explainDL/training/trainer.py
"""
trainer.py
----------
Unified training module for ExplainDL.

This module provides:
- Clean training loop for tabular, image, text data
- EarlyStopping & ReduceLROnPlateau
- Optional class weights
- Proper metric computation
- Full compatibility with:
    - sparse categorical multiclass classification
    - binary classification (sigmoid)
    - Keras generators (image mode)

Returned:
    history: dict of training curves
    metrics: dict of accuracy, precision, recall, f1, confusion_matrix, etc.
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from explainDL.training.metrics import compute_metrics


def compute_class_weights(y):
    """
    Computes class weights for imbalanced datasets.
    Works for integer labels.
    """
    unique = np.unique(y)
    if len(unique) <= 1:
        return None  # no weighting possible

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique,
        y=y
    )
    return {cls: w for cls, w in zip(unique, weights)}


def train_model(
    model,
    data_type,
    data,
    epochs=12,
    batch_size=32,
    use_class_weights=True
):
    """
    Unified training function used by pipeline_train.py.

    Parameters
    ----------
    model : keras.Model
    data_type : str
        'tabular', 'image', or 'text'
    data :
        TABULAR/TEXT → (X, y)
        IMAGE        → (train_gen, val_gen)
    epochs : int
    batch_size : int
    use_class_weights : bool

    Returns
    -------
    history : dict
    metrics : dict
    """

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # -----------------------------
    # IMAGE TRAINING (Generators)
    # -----------------------------
    if data_type == "image":
        train_gen, val_gen = data

        class_weights = None
        if use_class_weights:
            y = train_gen.classes
            w = compute_class_weights(y)
            if w is not None:
                class_weights = w

        history_obj = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        history = history_obj.history

        # VALIDATION LABELS
        y_true = val_gen.classes

        # PREDICTIONS
        y_pred_probs = model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

    # ------------------------------------
    # TABULAR & TEXT TRAINING (arrays)
    # ------------------------------------
    else:
        X, y = data

        class_weights = None
        if use_class_weights:
            w = compute_class_weights(y)
            if w is not None:
                class_weights = w

        history_obj = model.fit(
            X,
            y,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        history = history_obj.history

        # Predictions on validation split OR full X (safer)
        y_pred_probs = model.predict(X)

        # Convert to class indices
        if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = y
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            y_true = y

    # ------------------------------------
    # METRIC COMPUTATION
    # ------------------------------------
    metrics = compute_metrics(y_true, y_pred)

    return history, metrics
