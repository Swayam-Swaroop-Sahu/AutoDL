"""
text_loader.py
---------------
Handles loading of text datasets from plain text files.
"""

def load_text_data(file_path: str):
    """
    Loads raw text data from a .txt file.

    Parameters
    ----------
    file_path : str
        Path to the text file.

    Returns
    -------
    list[str]
        List of lines or text entries.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines
