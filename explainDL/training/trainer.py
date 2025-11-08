"""
trainer.py
-----------
Unified training routine for ExplainDL with improved callbacks (EarlyStopping)
and consistent handling of binary/multi-class outputs.
"""

import numpy as np
from explainDL.training.metrics import evaluate_predictions, summarize_results
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_and_evaluate(model, data_type, data, epochs=30, batch_size=32):
    """
    Trains a deep learning model and evaluates its performance.

    Parameters
    ----------
    model : keras.Model
    data_type : str
        'tabular', 'image', or 'text'
    data : tuple
        - For tabular/text: (X_train, X_test, y_train, y_test) where y may be one-hot for multi-class
        - For image: (train_gen, val_gen)
    epochs : int
    batch_size : int

    Returns
    -------
    dict
    """
    history = None
    metrics = None
    summary_df = None

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    if data_type == "image":
        train_gen, val_gen = data
        history_obj = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)
        history = history_obj.history

        # Predictions and ground truth from generator
        y_true = val_gen.classes
        y_pred_proba = model.predict(val_gen)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        metrics = evaluate_predictions(y_true, y_pred)
        summary_df = summarize_results(metrics)

    else:
        X_train, X_test, y_train, y_test = data

        history_obj = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
        history = history_obj.history

        y_pred_proba = model.predict(X_test)
        # Convert model outputs to class labels
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # If y_test is one-hot, convert to label indices
        if isinstance(y_test, np.ndarray) and y_test.ndim == 2 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = np.array(y_test).astype(int)

        metrics = evaluate_predictions(y_true, y_pred)
        summary_df = summarize_results(metrics)

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "summary": summary_df
    }
