# explainDL/model_selection/selector.py

import numpy as np
from tensorflow.keras.models import Model

from .tabular_models import build_mlp_small, build_mlp_medium, build_mlp_large
from .image_models import build_small_cnn, build_mobilenet, build_efficientnet
from .text_models import build_lstm, build_bilstm, build_text_cnn


def _safe_predict(model, sample_input):
    try:
        return model.predict(sample_input, verbose=0)
    except Exception:
        return None


def _quick_score(model, sample_input):
    out = _safe_predict(model, sample_input)
    if out is None:
        return float("-inf")
    try:
        return float(np.var(out))
    except Exception:
        return float("-inf")


def select_best_model(data_type: str, input_shape, num_classes: int):
    """
    Chooses best model from small set based on quick output variance.
    """

    candidates = []
    sample_input = None

    # ==================================================================
    # TABULAR
    # ==================================================================
    if data_type == "tabular":
        num_features = int(input_shape)

        candidates = [
            ("MLP-Small", build_mlp_small(num_features, num_classes)),
            ("MLP-Medium", build_mlp_medium(num_features, num_classes)),
            ("MLP-Large", build_mlp_large(num_features, num_classes)),
        ]

        sample_input = np.random.rand(8, num_features).astype("float32")

    # ==================================================================
    # IMAGE
    # ==================================================================
    elif data_type == "image":

        # Ensure (H, W, C)
        if len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 3)

        if len(input_shape) != 3 or input_shape[2] != 3:
            raise ValueError(
                f"Expected image input_shape=(H,W,3), received {input_shape}"
            )

        candidates = [
            ("Small-CNN", build_small_cnn(input_shape, num_classes)),
            ("MobileNetV2", build_mobilenet(input_shape, num_classes)),
            ("EfficientNetB0", build_efficientnet(input_shape, num_classes)),
        ]

        sample_input = np.random.rand(4, *input_shape).astype("float32")

    # ==================================================================
    # TEXT
    # ==================================================================
    elif data_type == "text":
        vocab_size, max_len = input_shape

        candidates = [
            ("BiLSTM", build_bilstm(vocab_size, max_len, num_classes)),
            ("LSTM", build_lstm(vocab_size, max_len, num_classes)),
            ("Text-CNN", build_text_cnn(vocab_size, max_len, num_classes)),
        ]

        sample_input = np.random.randint(1, max(2, vocab_size), size=(4, max_len))

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    # ==================================================================
    # SCORE MODELS
    # ==================================================================
    best_score = float("-inf")
    best_model = None
    best_name = None

    for name, model in candidates:
        score = _quick_score(model, sample_input)
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    if best_model is None:
        raise RuntimeError("Model selection failed — no valid candidates.")

    return best_model, best_name
