# src/utils/save_utils.py
"""
Handles model saving, loading, registry management, and preprocessing save/load.
Used by pipeline_train.py and pipeline_predict.py.
"""

import os
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model

from .file_utils import ensure_dir, write_json, read_json


def create_model_dir(base_dir="model_registry"):
    """
    Creates a new folder with timestamp-based unique ID.
    Returns full path.
    """
    ensure_dir(base_dir)
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_dir, model_id)
    ensure_dir(model_dir)
    return model_dir


def save_preprocessor(preprocessor, save_path: str):
    """Save preprocessors using pickle."""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(path: str):
    """Load saved preprocessor from pickle."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Preprocessor not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_keras(model, save_path="model.h5"):
    """Save a Keras model (.h5 or .keras)."""
    ensure_dir(os.path.dirname(save_path))
    model.save(save_path)


def load_keras_model(path: str):
    """Load Keras model from .h5 or .keras."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)


def save_metadata(meta: dict, save_path: str):
    """Write metadata JSON inside model directory."""
    write_json(meta, save_path)


def load_metadata(path: str):
    return read_json(path)


def get_model_paths(model_dir: str):
    """
    Utility to return likely paths inside a saved model directory.
    """
    return {
        "model": os.path.join(model_dir, "model.h5"),
        "preprocessor": os.path.join(model_dir, "preprocessor.pkl"),
        "metadata": os.path.join(model_dir, "meta.json"),
        "train_report": os.path.join(model_dir, "train_report.pdf"),
        "predict_report": os.path.join(model_dir, "predict_report.pdf"),
    }
