"""
Hyperparameter tuning utilities for ExplainDL.

Supports:
- Tabular (MLP)
- Image (small CNN)
- Text (Embedding + LSTM/CNN)

Now extended to support user-controlled hyperparameters via tuning_config.
"""

import os
import json
import traceback

try:
    import keras_tuner as kt
except Exception:
    kt = None

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers


# -----------------------------------------------------------------------------
# BUILDERS WITH USER-CONTROLLED SEARCH SPACES
# -----------------------------------------------------------------------------


def _build_tunable_mlp(hp, input_shape, num_classes, tuning_config):
    """
    Tunable MLP for tabular data.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    # Number of hidden layers
    n_layers = hp.Int("n_layers", 1, 3) if tuning_config["tune_hidden_layers"] else 2

    for i in range(n_layers):

        # Units
        if tuning_config["tune_units"]:
            units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
        else:
            units = 128

        model.add(layers.Dense(units, activation="relu"))

        # Dropout
        if tuning_config["tune_dropout"]:
            drop = hp.Float(f"dropout_{i}", 0.0, 0.6, step=0.1)
            if drop > 0:
                model.add(layers.Dropout(drop))

    # Output layer
    # BUGFIX Phase 1c: removed `if num_classes == 2` branch.
    loss = "sparse_categorical_crossentropy"
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Learning rate
    lr = (
        hp.Float("lr", 1e-4, 1e-2, sampling="log")
        if tuning_config["tune_learning_rate"]
        else 1e-3
    )

    model.compile(optimizer=optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    return model


def _build_tunable_cnn(hp, input_shape, num_classes, tuning_config):
    """
    Tunable CNN for image data.
    """
    inp = layers.Input(shape=input_shape)
    x = inp

    # Number of CNN blocks
    n_blocks = hp.Int("n_blocks", 1, 3) if tuning_config["tune_conv_blocks"] else 2

    for b in range(n_blocks):

        # Filters
        if tuning_config["tune_filters"]:
            filters = hp.Int(f"filters_{b}", 16, 128, step=16)
        else:
            filters = 32

        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)

        if tuning_config["tune_dropout"]:
            drop = hp.Float(f"drop_{b}", 0.0, 0.5, step=0.1)
            if drop > 0:
                x = layers.Dropout(drop)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer
    dense_units = (
        hp.Int("dense_units", 32, 256, step=32)
        if tuning_config["tune_units"]
        else 128
    )
    x = layers.Dense(dense_units, activation="relu")(x)

    # Output
    # BUGFIX Phase 1c: removed `if num_classes == 2` branch.
    loss = "sparse_categorical_crossentropy"
    out = layers.Dense(num_classes, activation="softmax")(x)

    lr = (
        hp.Float("lr", 1e-4, 1e-2, sampling="log")
        if tuning_config["tune_learning_rate"]
        else 1e-3
    )

    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    return model


def _build_tunable_text(hp, vocab_size, max_len, num_classes, tuning_config):
    """
    Tunable NLP model for text classification.
    """

    inp = layers.Input(shape=(max_len,))

    # Embedding dimension
    embed_dim = (
        hp.Int("embed_dim", 32, 256, step=32)
        if tuning_config["tune_embedding"]
        else 128
    )
    x = layers.Embedding(vocab_size, embed_dim)(inp)

    # Architecture selection
    arch = hp.Choice("arch", ["lstm", "bilstm", "cnn"]) if tuning_config["tune_arch"] else "lstm"

    if arch == "lstm":
        units = hp.Int("lstm_units", 32, 256, step=32) if tuning_config["tune_units"] else 128
        x = layers.LSTM(units)(x)

    elif arch == "bilstm":
        units = hp.Int("bilstm_units", 32, 256, step=32) if tuning_config["tune_units"] else 128
        x = layers.Bidirectional(layers.LSTM(units))(x)

    else:
        filters = hp.Int("cnn_filters", 32, 256, step=32) if tuning_config["tune_filters"] else 128
        x = layers.Conv1D(filters, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)

    # Dense Layer
    dense_units = hp.Int("dense_units", 32, 256, step=32) if tuning_config["tune_units"] else 128
    x = layers.Dense(dense_units, activation="relu")(x)

    # Output
    # BUGFIX Phase 1c: removed `if num_classes == 2` branch.
    out = layers.Dense(num_classes, activation="softmax")(x)
    loss = "sparse_categorical_crossentropy"

    # Learning rate
    lr = (
        hp.Float("lr", 1e-4, 1e-2, sampling="log")
        if tuning_config["tune_learning_rate"]
        else 1e-3
    )

    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    return model


# -----------------------------------------------------------------------------
# Convert Hyperparameters Object to Python dict
# -----------------------------------------------------------------------------

def hp_to_dict(hp):
    # BUGFIX Phase 1e item 3: replaced silent bare except with structured logging.
    try:
        return hp.get_config()
    except Exception as exc:
        from src.utils.logger import get_logger
        get_logger(__name__).warning(
            "BUGFIX Phase 1e item 3: hp.get_config() failed (%s); falling back to dict(hp.values).",
            exc,
        )
        try:
            return dict(hp.values)
        except Exception as exc2:
            get_logger(__name__).warning(
                "BUGFIX Phase 1e item 3: dict(hp.values) also failed (%s); returning empty dict.", exc2,
            )
            return {}


# -----------------------------------------------------------------------------
# FULL TUNING FUNCTION
# -----------------------------------------------------------------------------

def tune_model(
    model_type,
    train_data,
    input_shape,
    num_classes,
    tuning_config=None,
    max_trials=10,
    executions_per_trial=1,
    directory="tuning_logs",
    project_name="explaindl_tune",
    epochs=10,
    objective="val_accuracy",
):
    """
    Runs hyperparameter tuning and returns:
        { "best_model", "best_hyperparameters", "tuner" }

    tuning_config controls which hyperparameters are *allowed* to vary.
    """

    if kt is None:
        raise ImportError("keras_tuner not installed. Install using: pip install keras-tuner")

    if tuning_config is None:
        tuning_config = {
            "tune_learning_rate": True,
            "tune_units": True,
            "tune_hidden_layers": True,
            "tune_dropout": True,
            "tune_embedding": True,
            "tune_arch": True,
            "tune_filters": True,
            "tune_conv_blocks": True,
        }

    # Choose builder
    def build_model(hp):
        if model_type == "tabular":
            return _build_tunable_mlp(hp, input_shape, num_classes, tuning_config)
        elif model_type == "image":
            return _build_tunable_cnn(hp, input_shape, num_classes, tuning_config)
        elif model_type == "text":
            vocab_size, max_len = input_shape
            return _build_tunable_text(hp, vocab_size, max_len, num_classes, tuning_config)
        else:
            raise ValueError("Unsupported model_type for tuning")

    tuner = kt.RandomSearch(
        build_model,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name,
        overwrite=True,
    )

    # callbacks
    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    # Run search based on data type
    try:
        if model_type in ("tabular", "text"):
            X, y = train_data
            tuner.search(
                np.array(X),
                np.array(y),
                validation_split=0.15,
                epochs=epochs,
                callbacks=callback_list,
            )
        else:  # image
            train_gen, val_gen = train_data
            tuner.search(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callback_list,
            )

        # get best result
        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.hypermodel.build(best_hp)

        # retrain quickly
        if model_type in ("tabular", "text"):
            X, y = train_data
            best_model.fit(np.array(X), np.array(y), epochs=epochs, validation_split=0.1, verbose=0)
        else:
            train_gen, val_gen = train_data
            best_model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=0)

        return {
            "best_model": best_model,
            "best_hyperparameters": best_hp.values,
            "tuner": tuner
        }

    except Exception as e:
        raise RuntimeError(f"Tuning failed: {e}\n{traceback.format_exc()}")
