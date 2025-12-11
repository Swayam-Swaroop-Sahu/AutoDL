# explainDL/data/image_loader.py

import os
import zipfile
from PIL import Image


def extract_image_dataset(zip_path: str, extract_dir: str) -> str:
    """
    Extract ZIP and ensure it contains class-name subfolders.
    If no class folders exist, create a default folder.
    """

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Check extracted structure
    subfolders = [d for d in os.listdir(extract_dir)
                  if os.path.isdir(os.path.join(extract_dir, d))]

    # CASE: flat files (no class subfolders found)
    image_files = [f for f in os.listdir(extract_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(subfolders) == 0 and len(image_files) > 0:
        class_dir = os.path.join(extract_dir, "class_0")
        os.makedirs(class_dir, exist_ok=True)

        # move images into class_0
        for f in image_files:
            os.rename(os.path.join(extract_dir, f), os.path.join(class_dir, f))

    return extract_dir



def verify_images(image_dir: str) -> int:
    """
    Verifies all images by attempting to open them.

    Returns:
        count of valid images
    """

    valid = 0

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, f)
                try:
                    Image.open(path)
                    valid += 1
                except Exception:
                    pass

    return valid


def list_images(image_dir: str):
    """
    Returns a flat list of all image files inside image_dir.
    """

    paths = []

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))

    return paths
