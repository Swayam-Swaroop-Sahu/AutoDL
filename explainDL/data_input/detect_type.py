"""
detect_type.py
---------------
Automatically detects the dataset type (tabular, image, or text)
based on the uploaded file’s content and structure.
"""

import os
import zipfile
import pandas as pd

def detect_dataset_type(file_path: str) -> str:
    """
    Detects dataset type based on file extension and content.
    
    Parameters
    ----------
    file_path : str
        Path to the uploaded file.

    Returns
    -------
    str
        'tabular', 'image', or 'text'
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
            # Simple heuristic: if text-like columns dominate, classify later
            text_columns = df.select_dtypes(include="object").shape[1]
            if text_columns > 0.7 * df.shape[1]:
                return "text"
            return "tabular"
        except Exception:
            return "tabular"

    elif ext == ".zip":
        # Image datasets are often zipped folders of image files
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                names = zip_ref.namelist()
                image_files = [n for n in names if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if len(image_files) > 0.5 * len(names):
                    return "image"
        except Exception:
            pass
        return "image"

    elif ext == ".txt":
        return "text"

    else:
        return "tabular"
