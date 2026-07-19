# src/core/model_registry.py

import os
import json
import joblib
from tensorflow.keras.models import load_model

from src.core.config import MODEL_REGISTRY_DIR


def load_saved_model(model_dir: str):
    """Loads model + preprocessor + metadata."""

    model = load_model(os.path.join(model_dir, "model.h5"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))

    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, preprocessor, meta
