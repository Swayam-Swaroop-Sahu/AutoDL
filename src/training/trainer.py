# src/training/trainer.py
"""
trainer.py
----------
Unified training module for ExplainDL.

BUGFIX Phase 1e item 13: model comparison uses 3-fold stratified CV during
search; this module's `train_model` then fits the winner on full data with
a held-out validation split (the validation metrics reported in meta.json
are honest held-out scores; CV scores live in `search_results`).


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
from sklearn.model_selection import train_test_split

from src.training.metrics import compute_metrics


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

        # VALIDATION LABELS (one-hot from generator) → integer class indices
        y_true_onehot = val_gen.classes
        # BUGFIX Phase 1c: val_gen.classes is already integer indices for categorical.
        # (image generator with class_mode="categorical" returns indices via .classes.)
        y_true = np.asarray(y_true_onehot, dtype=int)

        # PREDICTIONS
        y_pred_probs = model.predict(val_gen, verbose=0)
        # BUGFIX Phase 1c: always argmax; softmax output is shape (N, num_classes).
        y_pred = np.argmax(y_pred_probs, axis=1)

    # ------------------------------------
    # TABULAR & TEXT TRAINING (arrays)
    # ------------------------------------
    else:
        X, y = data
        X = np.asarray(X)
        y = np.asarray(y)

        if len(y) < 10:
            raise ValueError("At least 10 labelled samples are required to create a reliable validation split.")

        _, class_counts = np.unique(y, return_counts=True)
        stratify = y if np.all(class_counts >= 2) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        class_weights = None
        if use_class_weights:
            w = compute_class_weights(y_train)
            if w is not None:
                class_weights = w

        history_obj = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        history = history_obj.history

        y_pred_probs = model.predict(X_val, verbose=0)

        # BUGFIX Phase 1c: always argmax; softmax output is shape (N, num_classes).
        y_pred = np.argmax(y_pred_probs, axis=1)

        y_true = y_val

    # ------------------------------------
    # METRIC COMPUTATION
    # ------------------------------------
    metrics = compute_metrics(y_true, y_pred)
    # These arrays allow reports to render the same held-out evaluation result.
    metrics["y_true"] = np.asarray(y_true, dtype=int).tolist()
    metrics["y_pred"] = np.asarray(y_pred, dtype=int).tolist()

    return history, metrics
