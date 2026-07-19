# src/registry/registry.py
"""
Simple, reliable Model Registry for ExplainDL.

Keeps a central index (model_index.json) that records:
- model_dir
- model_name
- dataset_type
- metrics
- timestamp

This registry does NOT save model/preprocessor files.
Those are handled by pipeline_train using utils/save_utils.py.
"""

import os
import json
from datetime import datetime

from src.utils.file_utils import ensure_dir
from src.utils.save_utils import load_metadata


REGISTRY_DIR = "model_registry"
REGISTRY_INDEX = os.path.join(REGISTRY_DIR, "model_index.json")


# --------------------------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------------------------
def _ensure_index():
    """Ensure registry folder + index file exist."""
    ensure_dir(REGISTRY_DIR)
    if not os.path.exists(REGISTRY_INDEX):
        with open(REGISTRY_INDEX, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def _load_index() -> dict:
    _ensure_index()
    with open(REGISTRY_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: dict):
    ensure_dir(os.path.dirname(REGISTRY_INDEX))
    with open(REGISTRY_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# --------------------------------------------------------------------
# PUBLIC FUNCTIONS
# --------------------------------------------------------------------
def register_model(model_dir: str):
    """
    Adds a trained model directory into registry index.
    Metadata is read from model_dir/meta.json.
    """
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot register model — meta.json missing in {model_dir}")

    metadata = load_metadata(meta_path)
    index = _load_index()

    entry = {
        "model_dir": model_dir,
        "model_name": metadata.get("model_name", "Unknown"),
        "dataset_type": metadata.get("dataset_type", "Unknown"),
        "metrics": metadata.get("metrics", {}),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    index[model_dir] = entry
    _save_index(index)

    return entry


def list_models() -> list:
    """
    Return list of all model registry entries.
    """
    return list(_load_index().values())


def get_model_entry(model_dir: str) -> dict:
    """Return metadata entry for a specific saved model."""
    index = _load_index()
    return index.get(model_dir)
