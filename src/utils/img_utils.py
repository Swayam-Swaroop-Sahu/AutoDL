"""
Image-related utility functions.

These are lightweight helpers used by loaders, validation,
and sometimes explainability modules.
"""

import os
from typing import Optional

from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


def is_image_file(path: str) -> bool:
    """
    Checks if a file path looks like an image based on its extension.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    """
    return path.lower().endswith(IMAGE_EXTENSIONS)


def load_image_safe(path: str) -> Optional[Image.Image]:
    """
    Safely loads an image. Returns None if the file is not a valid image.

    Parameters
    ----------
    path : str

    Returns
    -------
    PIL.Image.Image or None
    """
    if not os.path.isfile(path):
        return None

    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None
