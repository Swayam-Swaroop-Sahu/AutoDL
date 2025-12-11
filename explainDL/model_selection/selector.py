# explainDL/model_selection/selector.py
"""
Model selection helper.

Provides:
- select_best_model(data_type, input_shape, num_classes)
  -> returns (keras.Model, model_name)

This file evaluates a small set of candidate architectures for each data
type using a quick forward-pass "sanity" check and a lightweight variance
score to pick a candidate automatically.
"""

import numpy as np
import traceback

# Keras is imported here only when needed to avoid heavy import on module load.
from tensorflow.keras.models import Model

from .tabular_models import build_mlp_small, build_mlp_medium, build_mlp_large
from .image_models import build_small_cnn, build_mobilenet, build_efficientnet
from .text_models import build_lstm, build_bilstm, build_text_cnn



def _safe_predict(model, sample_input):
    """
    Run a single forward pass safely. If it fails, return None.
    """
    try:
        # Some models expect int input (text embedding indices)
        out = model.predict(sample_input, verbose=0)
        return out
    except Exception:
        # attempt to trace error for debugging (non-blocking)
        # print(traceback.format_exc())
        return None


def _quick_score(model, sample_input):
    """
    Compute a quick numeric score from a forward pass that helps us compare models.
    The score is variance of the output tensor (higher = more spread = "active").
    Returns -inf on error.
    """
    out = _safe_predict(model, sample_input)
    if out is None:
        return float("-inf")
    try:
        arr = np.array(out)
        # If output is all identical values, var will be 0 — small score
        return float(np.var(arr))
    except Exception:
        return float("-inf")


def select_best_model(data_type: str, input_shape, num_classes: int):
    """
    Selects the best candidate model for the given data_type.

    Parameters
    ----------
    data_type : str
        One of "tabular", "image", "text"
    input_shape :
        - tabular: integer (num_features)
        - image: tuple (H, W, C)
        - text: tuple (vocab_size, max_len)
    num_classes : int
        Number of classes (>=2)

    Returns
    -------
    (model, model_name)
    """

    candidates = []
    sample_input = None

    # -------------------------
    # TABULAR
    # -------------------------
    if data_type == "tabular":
        # Expect integer for number of features
        num_features = int(input_shape)
        candidates = [
            ("MLP-Small", build_mlp_small(num_features, num_classes)),
            ("MLP-Medium", build_mlp_medium(num_features, num_classes)),
            ("MLP-Large", build_mlp_large(num_features, num_classes)),
        ]
        sample_input = np.random.rand(8, num_features).astype("float32")

    # -------------------------
    # IMAGE
    # -------------------------
    elif data_type == "image":
        # Expect input_shape like (H, W, C)
        candidates = [
            ("Small-CNN", build_small_cnn(input_shape, num_classes)),
            ("MobileNetV2", build_mobilenet(input_shape, num_classes)),
            ("EfficientNetB0", build_efficientnet(input_shape, num_classes)),
        ]
        # Small batch of random images
        sample_input = np.random.rand(4, *input_shape).astype("float32")
    elif data_type == "image":
        # ---------------------------
        # REQUIRED INPUT VALIDATION
        # ---------------------------
        if len(input_shape) == 2:
            # convert (H, W) → (H, W, 3)
            input_shape = (input_shape[0], input_shape[1], 3)

        if len(input_shape) != 3 or input_shape[2] != 3:
            raise ValueError(
                f"Image models require input shape = (H, W, 3). Received: {input_shape}. "
                "Fix your preprocessor so images are RGB."
            )

        # ---------------------------
        # CREATE CANDIDATE MODELS
        # ---------------------------
        candidates = [
            ("Small-CNN", build_small_cnn(input_shape, num_classes)),
            ("MobileNetV2", build_mobilenet(input_shape, num_classes)),
            ("EfficientNetB0", build_efficientnet(input_shape, num_classes)),
        ]

        # sample batch for quick score
        sample_input = np.random.rand(4, *input_shape).astype("float32")


    # -------------------------
    # TEXT
    # -------------------------
    elif data_type == "text":
        # Expect (vocab_size, max_len)
        vocab_size, max_len = input_shape
        candidates = [
            ("BiLSTM", build_bilstm(vocab_size, max_len, num_classes)),
            ("LSTM", build_lstm(vocab_size, max_len, num_classes)),
            ("Text-CNN", build_text_cnn(vocab_size, max_len, num_classes)),
        ]
        sample_input = np.random.randint(1, max(2, vocab_size), size=(4, max_len))

    else:
        raise ValueError(f"Unsupported data_type for model selection: {data_type}")

    # -------------------------
    # Evaluate candidates with quick score
    # -------------------------
    best_score = float("-inf")
    best_model = None
    best_name = None

    for name, model in candidates:
        try:
            score = _quick_score(model, sample_input)
            # prefer non-error and higher variance
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception:
            # If any model creation/prediction fails, continue
            continue

    if best_model is None:
        raise RuntimeError("Model selection failed — no candidate model passed the sanity check.")

    return best_model, best_name
