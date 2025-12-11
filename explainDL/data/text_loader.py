# explainDL/data/text_loader.py

def load_text_file(file_path: str):
    """
    Loads a TXT file and returns a list of non-empty lines.
    Preserves order.
    """

    lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                lines.append(clean)

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
    """

    texts = []
    labels = []

    for line in lines:
        # TAB separated
        if "\t" in line:
            label, text = line.split("\t", 1)
            labels.append(label.strip())
            texts.append(text.strip())
            continue

        # COMMA separated
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2:
                label, text = parts
                labels.append(label.strip())
                texts.append(text.strip())
                continue

        # Not labelled → prediction mode
        return None, None

    return texts, labels
