"""
image_loader.py
----------------
Handles extraction and loading of image datasets from a ZIP file.
"""

import zipfile
import os
from PIL import Image
from io import BytesIO

def extract_image_dataset(zip_path: str, extract_dir: str = "temp_images") -> str:
    """
    Extracts image dataset from a ZIP file.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP file containing images.
    extract_dir : str, optional
        Directory to extract images into.

    Returns
    -------
    str
        Path to extracted image folder.
    """
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def verify_images(image_dir: str) -> int:
    """
    Verifies image integrity by attempting to open each file.

    Parameters
    ----------
    image_dir : str
        Path to folder containing images.

    Returns
    -------
    int
        Number of valid image files.
    """
    valid_count = 0
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    Image.open(os.path.join(root, f))
                    valid_count += 1
                except Exception:
                    pass
    return valid_count
