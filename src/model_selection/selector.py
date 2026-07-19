"""Deterministic baseline selection for supported classification datasets."""

from .tabular_models import build_mlp_small, build_mlp_medium, build_mlp_large
from .image_models import build_small_cnn, build_mobilenet, build_efficientnet
from .text_models import build_lstm, build_bilstm, build_text_cnn


def _recommend_model(data_type: str, input_shape, n_samples: int | None) -> str:
    """Choose a safe baseline from data scale without pretending to measure accuracy."""
    n_samples = n_samples or 0
    if data_type == "tabular":
        n_features = int(input_shape)
        if n_samples <= 500 or n_features <= 20:
            return "MLP-Small"
        return "MLP-Medium" if n_features <= 100 else "MLP-Large"
    if data_type == "image":
        if n_samples < 500:
            return "Small-CNN"
        return "MobileNetV2" if n_samples < 5_000 else "EfficientNetB0"
    if data_type == "text":
        return "Text-CNN" if n_samples < 500 else "BiLSTM"
    raise ValueError(f"Unsupported data_type: {data_type}")


def select_best_model(data_type: str, input_shape, num_classes: int, manual_selection: str = None,
                      n_samples: int | None = None):
    """Build one selected candidate and return transparent selection metadata.

    This function intentionally does not rank untrained random networks. Real model
    comparison requires identical validation folds and is a separate AutoML stage.
    """
    if data_type == "tabular":
        n_features = int(input_shape)
        candidates = [
            ("MLP-Small", lambda: build_mlp_small(n_features, num_classes)),
            ("MLP-Medium", lambda: build_mlp_medium(n_features, num_classes)),
            ("MLP-Large", lambda: build_mlp_large(n_features, num_classes)),
        ]
        descriptions = {
            "MLP-Small": ("Lightweight MLP with 64 hidden units.", "~4K", "Fast baseline", "Limited capacity"),
            "MLP-Medium": ("Balanced MLP with batch normalization.", "~15K", "Balanced capacity", "Moderate training time"),
            "MLP-Large": ("Higher-capacity MLP for wider feature sets.", "~50K", "More capacity", "Greater overfitting risk"),
        }
    elif data_type == "image":
        if len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 3)
        if len(input_shape) != 3 or input_shape[2] != 3:
            raise ValueError(f"Expected image input_shape=(H, W, 3), received {input_shape}")
        candidates = [
            ("Small-CNN", lambda: build_small_cnn(input_shape, num_classes)),
            ("MobileNetV2", lambda: build_mobilenet(input_shape, num_classes)),
            ("EfficientNetB0", lambda: build_efficientnet(input_shape, num_classes)),
        ]
        descriptions = {
            "Small-CNN": ("Custom CNN trained from scratch.", "~200K", "No external weights", "Needs more data"),
            "MobileNetV2": ("Efficient transfer-learning CNN.", "~3.5M", "Pretrained features", "More memory"),
            "EfficientNetB0": ("Larger efficient transfer-learning CNN.", "~5M", "Strong capacity", "Slower inference"),
        }
    elif data_type == "text":
        vocab_size, max_len = input_shape
        candidates = [
            ("BiLSTM", lambda: build_bilstm(vocab_size, max_len, num_classes)),
            ("LSTM", lambda: build_lstm(vocab_size, max_len, num_classes)),
            ("Text-CNN", lambda: build_text_cnn(vocab_size, max_len, num_classes)),
        ]
        descriptions = {
            "BiLSTM": ("Bidirectional sequence model.", "~100K", "Longer context", "Slower training"),
            "LSTM": ("Single-direction sequence baseline.", "~50K", "Simple sequence model", "Forward context only"),
            "Text-CNN": ("Convolutional text classifier.", "~80K", "Fast on small datasets", "Limited long-range context"),
        }
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    recommended = _recommend_model(data_type, input_shape, n_samples)
    names = {name for name, _ in candidates}
    if manual_selection and manual_selection not in names:
        raise ValueError(f"Manual selection '{manual_selection}' not found. Available models: {sorted(names)}")
    selected = manual_selection or recommended

    comparison = {"models": [], "selected": selected, "reason": ""}
    for name, _ in candidates:
        description, params, pros, cons = descriptions[name]
        comparison["models"].append({
            "name": name,
            "score": 1.0 if name == recommended else 0.0,
            "description": description,
            "params": params,
            "pros": pros,
            "cons": cons,
        })
    if manual_selection:
        comparison["reason"] = f"Manually selected by user. Scale-based baseline recommendation: {recommended}."
    else:
        comparison["reason"] = (
            "Selected as a deterministic baseline from dataset type, feature shape, and sample count. "
            "The score is a recommendation flag, not an accuracy measurement."
        )

    factory = next(factory for name, factory in candidates if name == selected)
    return factory(), selected, comparison
