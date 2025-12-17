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


def select_best_model(data_type: str, input_shape, num_classes: int, manual_selection: str = None):
    """
    Chooses best model from small set based on quick output variance.
    Returns model, name, and comparison data.
    
    Args:
        data_type: "tabular", "image", or "text"
        input_shape: Shape of input data
        num_classes: Number of classes
        manual_selection: Optional model name to override automatic selection
    
    Returns:
        tuple: (best_model, best_name, comparison_data)
        comparison_data: dict with keys:
            - models: list of dicts with name, score, description, params
            - selected: name of selected model
            - reason: explanation for selection
    """

    candidates = []
    sample_input = None
    model_descriptions = {}

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

        model_descriptions = {
            "MLP-Small": {
                "description": "Lightweight MLP with 64 hidden units. Fast training, good for small datasets.",
                "params": "~4K parameters",
                "pros": "Fast, low memory, good baseline",
                "cons": "Limited capacity for complex patterns"
            },
            "MLP-Medium": {
                "description": "Medium MLP with 128→64 hidden layers. Balanced capacity and speed.",
                "params": "~15K parameters",
                "pros": "Good balance, batch normalization",
                "cons": "Moderate training time"
            },
            "MLP-Large": {
                "description": "Large MLP with 256→128→64 hidden layers. High capacity for complex patterns.",
                "params": "~50K parameters",
                "pros": "High capacity, good for complex data",
                "cons": "Slower training, risk of overfitting"
            }
        }

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

        model_descriptions = {
            "Small-CNN": {
                "description": "Custom CNN from scratch. Lightweight, trains quickly.",
                "params": "~200K parameters",
                "pros": "Fast training, no pretrained weights needed",
                "cons": "Limited feature extraction capability"
            },
            "MobileNetV2": {
                "description": "MobileNetV2 with ImageNet pretrained weights. Efficient and accurate.",
                "params": "~3.5M parameters (frozen backbone)",
                "pros": "Pretrained features, efficient architecture",
                "cons": "Requires more memory than Small-CNN"
            },
            "EfficientNetB0": {
                "description": "EfficientNetB0 with ImageNet pretrained weights. State-of-the-art efficiency.",
                "params": "~5M parameters (frozen backbone)",
                "pros": "Best accuracy/efficiency tradeoff, pretrained",
                "cons": "Largest model, slower inference"
            }
        }

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

        model_descriptions = {
            "BiLSTM": {
                "description": "Bidirectional LSTM. Captures context from both directions.",
                "params": "~100K parameters",
                "pros": "Best for sequential understanding, bidirectional",
                "cons": "Slower training than unidirectional"
            },
            "LSTM": {
                "description": "Single-direction LSTM. Good baseline for text classification.",
                "params": "~50K parameters",
                "pros": "Faster than BiLSTM, good sequential modeling",
                "cons": "Only forward context"
            },
            "Text-CNN": {
                "description": "CNN-based text classifier. Fast and effective for many tasks.",
                "params": "~80K parameters",
                "pros": "Fast training, good for local patterns",
                "cons": "Limited long-range dependencies"
            }
        }

        sample_input = np.random.randint(1, max(2, vocab_size), size=(4, max_len))

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    # ==================================================================
    # SCORE MODELS
    # ==================================================================
    comparison_data = {
        "models": [],
        "selected": None,
        "reason": ""
    }

    best_score = float("-inf")
    best_model = None
    best_name = None

    for name, model in candidates:
        score = _quick_score(model, sample_input)
        desc = model_descriptions.get(name, {})
        
        comparison_data["models"].append({
            "name": name,
            "score": float(score),
            "description": desc.get("description", ""),
            "params": desc.get("params", ""),
            "pros": desc.get("pros", ""),
            "cons": desc.get("cons", "")
        })
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    if best_model is None:
        raise RuntimeError("Model selection failed — no valid candidates.")

    # Handle manual override
    recommended_name = best_name
    recommended_score = best_score
    if manual_selection:
        manual_found = False
        for name, model in candidates:
            if name == manual_selection:
                best_model = model
                best_name = name
                manual_found = True
                comparison_data["selected"] = manual_selection
                comparison_data["reason"] = (
                    "Manually selected by user. "
                    f"Original auto recommendation was {recommended_name} (score: {recommended_score:.4f})."
                )
                break
        
        if not manual_found:
            raise ValueError(f"Manual selection '{manual_selection}' not found. Available models: {[name for name, _ in candidates]}")
    else:
        comparison_data["selected"] = best_name
        comparison_data["reason"] = f"Automatically selected based on output variance score ({best_score:.4f}). Higher variance indicates better initial feature separation."

    return best_model, best_name, comparison_data
