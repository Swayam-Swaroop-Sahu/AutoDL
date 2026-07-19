# src/utils/file_utils.py
"""
General file/directory utilities for ExplainDL.
Covers:
- safe directory creation
- safe file removal
- reading/writing JSON
- listing files
- flattening ZIP extraction path
"""

import os
import json
import shutil
import zipfile


def ensure_dir(path: str):
    """Create path if not exists."""
    os.makedirs(path, exist_ok=True)
    return path


def remove_dir(path: str):
    """Remove directory safely if it exists."""
    if path and os.path.exists(path):
        shutil.rmtree(path)


def remove_file(path: str):
    """Remove a file safely."""
    if path and os.path.isfile(path):
        os.remove(path)


def write_json(data, save_path: str):
    """Write dictionary to JSON."""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path: str):
    """Load JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_files_recursive(dir_path: str, extensions=None):
    """
    List all files in directory recursively.
    If extensions provided: filter by extension tuple.
    """
    paths = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if extensions:
                if f.lower().endswith(extensions):
                    paths.append(os.path.join(root, f))
            else:
                paths.append(os.path.join(root, f))
    return paths


def unzip_to_dir(zip_path: str, extract_dir: str):
    """
    Extracts ZIP archive to extract_dir.
    """
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir
