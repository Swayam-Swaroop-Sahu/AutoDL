# src/data/tabular_loader.py

import os
import pandas as pd


def load_tabular_data(file_path: str, require_target: bool = True) -> pd.DataFrame:
    """
    Loads a CSV or Excel file as a pandas DataFrame.

    Ensures:
    - Unnamed index columns removed
    - Fully empty rows removed
    - Basic validation applied

    Returns:
        pd.DataFrame
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or data is invalid
        pd.errors.EmptyDataError: If file is empty
        pd.errors.ParserError: If file cannot be parsed
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please check the file path.")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path}. Please provide a non-empty dataset.")

    ext = file_path.lower()

    try:
        if ext.endswith(".csv"):
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="warn")
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
                except pd.errors.EmptyDataError:
                    raise ValueError(f"CSV file is empty or contains no data: {file_path}")
                except pd.errors.ParserError as e:
                    raise ValueError(f"CSV file cannot be parsed. Error: {str(e)}. Please check the file format.")
            
            if df is None:
                raise ValueError(f"Could not read CSV file with any encoding. Last error: {str(last_error)}")

        elif ext.endswith(".xlsx"):
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                if "No columns to parse" in str(e) or isinstance(e, pd.errors.EmptyDataError):
                    raise ValueError(f"Excel file is empty or contains no data: {file_path}")
                elif "not supported" in str(e).lower():
                    raise ValueError(f"Excel file format not supported. Please ensure it's a valid .xlsx file: {file_path}")
                else:
                    raise ValueError(f"Could not read Excel file: {str(e)}. Please check if the file is corrupted.")

        else:
            raise ValueError(f"Unsupported tabular file format: {file_path}. Supported formats: .csv, .xlsx")

    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty or contains no data: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"File cannot be parsed. Error: {str(e)}. Please check the file format.")

    # Validate DataFrame
    if df is None:
        raise ValueError(f"Failed to load data from {file_path}")

    # Remove unnamed columns (common in CSV exports)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Validate minimum requirements
    if df.empty:
        raise ValueError(f"Tabular file is empty or invalid after cleaning: {file_path}. Please ensure the file contains data.")

    min_rows = 10 if require_target else 1
    if df.shape[0] < min_rows:
        purpose = "training" if require_target else "prediction"
        raise ValueError(f"Dataset has too few rows ({df.shape[0]}) for {purpose}. Minimum {min_rows} row(s) required.")

    min_columns = 2 if require_target else 1
    if df.shape[1] < min_columns:
        raise ValueError(f"Dataset has too few columns ({df.shape[1]}). Minimum {min_columns} column(s) required.")

    # Check for all-NaN columns
    nan_cols = df.columns[df.isna().all()].tolist()
    if nan_cols:
        raise ValueError(f"Dataset contains columns with all missing values: {nan_cols}. Please remove or fill these columns.")

    return df
