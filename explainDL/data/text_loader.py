# explainDL/data/text_loader.py

import os


def load_text_file(file_path: str):
    """
    Loads a TXT file and returns a list of non-empty lines.
    Preserves order.
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}. Please check the file path.")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"Text file is empty: {file_path}. Please provide a non-empty dataset.")

    lines = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    last_error = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    clean = line.strip()
                    if clean:
                        lines.append(clean)
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise ValueError(f"Could not read text file: {str(e)}")

    if len(lines) == 0 and last_error:
        raise ValueError(f"Could not read text file with any encoding. Last error: {str(last_error)}")

    if len(lines) == 0:
        raise ValueError(f"Text file contains no valid lines: {file_path}. Please ensure the file contains data.")

    if len(lines) < 2:
        raise ValueError(f"Text file has too few lines ({len(lines)}). Minimum 2 lines required for training.")

    return lines


def parse_labelled_text(lines):
    """
    Parses labelled text lines.

    Each line must be in one of the formats:
        label<TAB>text
        label,text

    Returns:
        texts: list[str]
        labels: list[str]

    If the file is not labelled → (None, None)
    
    Raises:
        ValueError: If format is invalid or data is insufficient
    """

    if not lines or len(lines) == 0:
        raise ValueError("No lines provided for parsing.")

    texts = []
    labels = []
    invalid_lines = []

    for idx, line in enumerate(lines, 1):
        # TAB separated
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                label = label.strip()
                text = text.strip()
                if label and text:
                    labels.append(label)
                    texts.append(text)
                else:
                    invalid_lines.append(idx)
            else:
                invalid_lines.append(idx)
            continue

        # COMMA separated
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2:
                label, text = parts
                label = label.strip()
                text = text.strip()
                if label and text:
                    labels.append(label)
                    texts.append(text)
                else:
                    invalid_lines.append(idx)
            else:
                invalid_lines.append(idx)
            continue

        # Not labelled → prediction mode (return None, None only if no valid labels found yet)
        if len(labels) == 0:
            return None, None
        else:
            # Mixed format - some lines have labels, some don't
            invalid_lines.append(idx)

    # Validate we have enough data
    if len(texts) == 0:
        raise ValueError("No valid labelled text lines found. Expected format: 'label<TAB>text' or 'label,text' per line.")

    if len(texts) < 2:
        raise ValueError(f"Too few valid text samples ({len(texts)}). Minimum 2 samples required for training.")

    if len(set(labels)) < 2:
        raise ValueError(f"Dataset has only {len(set(labels))} unique label(s). Minimum 2 different labels required for classification.")

    if invalid_lines:
        # Warn but don't fail if we have enough valid data
        if len(texts) >= 2:
            pass  # We have enough valid data, just log the issue
        else:
            raise ValueError(f"Too many invalid lines found (lines: {invalid_lines[:10]}). Expected format: 'label<TAB>text' or 'label,text'.")

    return texts, labels
