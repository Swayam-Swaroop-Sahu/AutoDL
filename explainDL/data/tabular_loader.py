# explainDL/data/tabular_loader.py

import pandas as pd


def load_tabular_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV or Excel file as a pandas DataFrame.

    Ensures:
    - Unnamed index columns removed
    - Fully empty rows removed
    - Basic validation applied

    Returns:
        pd.DataFrame
    """

    ext = file_path.lower()

    if ext.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif ext.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported tabular file format: {file_path}")

    # Remove unnamed columns (common in CSV exports)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    if df.empty:
        raise ValueError("Tabular file is empty or invalid after cleaning.")

    return df
