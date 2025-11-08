"""
tabular_loader.py
-----------------
Handles loading of tabular datasets (CSV, Excel).
"""

import pandas as pd

def load_tabular_data(file_path: str):
    """
    Loads a tabular dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to CSV or Excel file.

    Returns
    -------
    pandas.DataFrame
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format for tabular data.")
    return df
