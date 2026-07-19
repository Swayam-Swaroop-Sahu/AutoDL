from tensorflow.keras import layers, models


def _compile_top(model, num_classes):
    """
    Add the final classification layer to both Sequential and Functional models.
    Works for all text models.
    """

    # Sequential models → add directly
    if isinstance(model, models.Sequential):
        model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return model

    # Functional models → wrap with new head
    elif isinstance(model, models.Model):
        x = model.output
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        new_model = models.Model(inputs=model.input, outputs=outputs)

        new_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return new_model

    else:
        raise ValueError("Unsupported model type for text classification.")


# =========================================================
# SIMPLE LSTM MODEL  (OPTION B ADDED)
# =========================================================
def build_lstm(vocab_size, max_len, num_classes):
    """
    Simple single-layer LSTM classifier.
    Good baseline for comparison.
    """

    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, 128)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)

    base = models.Model(inputs, x)
    return _compile_top(base, num_classes)


# =========================================================
# BiLSTM MODEL
# =========================================================
def build_bilstm(vocab_size, max_len, num_classes):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, 128)(inputs)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation="relu")(x)

    base = models.Model(inputs, x)
    return _compile_top(base, num_classes)


# =========================================================
# CNN TEXT MODEL
# =========================================================
def build_text_cnn(vocab_size, max_len, num_classes):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, 128)(inputs)

    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)

    base = models.Model(inputs, x)
    return _compile_top(base, num_classes)
