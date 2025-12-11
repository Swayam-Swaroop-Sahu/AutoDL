# explainDL/data/detect_type.py

import os
import zipfile
import pandas as pd


def detect_dataset_type(file_path: str) -> str:
    """
    Detects dataset type based on:
    - file extension
    - file structure
    - content heuristics

    Returns:
        "tabular"
        "text"
        "image"
        "unknown"
    """

    ext = os.path.splitext(file_path)[1].lower()

    # ----------------------------------------------------
    # TABULAR (CSV, XLSX)
    # ----------------------------------------------------
    if ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)

            # If 70%+ columns are text → this is actually a TEXT dataset
            object_cols = df.select_dtypes(include="object").shape[1]
            if object_cols > 0.7 * df.shape[1]:
                return "text"

            return "tabular"

        except Exception:
            return "tabular"

    # ----------------------------------------------------
    # TEXT (TXT)
    # ----------------------------------------------------
    if ext == ".txt":
        return "text"

    # ----------------------------------------------------
    # IMAGE ZIP
    # ----------------------------------------------------
    if ext == ".zip":
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                names = z.namelist()

                if len(names) == 0:
                    return "unknown"

                image_files = [
                    n for n in names
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

                if len(image_files) == 0:
                    return "unknown"

                # At least 50% of files must be images to classify as IMAGE dataset
                if len(image_files) >= 0.5 * len(names):
                    return "image"

                return "unknown"

        except Exception:
            return "unknown"

    # ----------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------
    return "unknown"
