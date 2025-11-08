"""
file_utils.py
-------------
Handles file path management, saving/loading objects,
and temporary storage for ExplainDL.
"""

import os
import joblib
import json

def ensure_dir(path: str):
    """
    Ensures that a directory exists.
    """
    os.makedirs(path, exist_ok=True)


def save_object(obj, path: str):
    """
    Saves a Python object using joblib.
    """
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_object(path: str):
    """
    Loads a saved Python object.
    """
    return joblib.load(path)


def save_json(data: dict, path: str):
    """
    Saves a dictionary as a JSON file.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_json(path: str):
    """
    Reads a JSON configuration file.
    """
    with open(path, "r") as f:
        return json.load(f)
