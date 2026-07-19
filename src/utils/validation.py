"""
Dataset validation utilities.

These functions perform basic sanity checks on input data before
they are sent to preprocessing or training pipelines.
"""

import os
from typing import List, Tuple

import pandas as pd

from .img_utils import is_image_file, load_image_safe


def validate_tabular_dataframe(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> Tuple[bool, List[str]]:
    """
    Validates a tabular DataFrame for basic conditions:
    - minimum rows and columns
    - no completely empty columns

    Parameters
    ----------
    df : pandas.DataFrame
    min_rows : int
    min_cols : int

    Returns
    -------
    (is_valid, warnings)
    """
    warnings = []

    if df.shape[0] < min_rows:
        warnings.append(f"DataFrame has only {df.shape[0]} rows (min {min_rows} recommended).")
    if df.shape[1] < min_cols:
        warnings.append(f"DataFrame has only {df.shape[1]} columns (min {min_cols} recommended).")

    # Check for columns with all NaNs
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        warnings.append(f"The following columns are completely empty: {empty_cols}")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def validate_image_dir(image_dir: str, min_images: int = 10) -> Tuple[bool, List[str]]:
    """
    Validates that a directory contains enough valid image files.

    Parameters
    ----------
    image_dir : str
        Root directory containing image dataset (subfolders = classes).
    min_images : int
        Minimum total images required.

    Returns
    -------
    (is_valid, warnings)
    """
    warnings = []

    if not os.path.isdir(image_dir):
        return False, [f"Image directory not found: {image_dir}"]

    total_images = 0
    invalid_images = 0

    for root, _, files in os.walk(image_dir):
        for f in files:
            path = os.path.join(root, f)
            if not is_image_file(path):
                continue
            img = load_image_safe(path)
            if img is None:
                invalid_images += 1
            else:
                total_images += 1

    if total_images < min_images:
        warnings.append(f"Only {total_images} valid images found (min {min_images} recommended).")

    if invalid_images > 0:
        warnings.append(f"{invalid_images} files could not be read as valid images.")

    is_valid = len(warnings) == 0
    return is_valid, warnings
