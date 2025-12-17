# explainDL/data/image_loader.py

import os
import zipfile
from PIL import Image


def extract_image_dataset(zip_path: str, extract_dir: str) -> str:
    """
    Extract ZIP and ensure it contains class-name subfolders.
    If no class folders exist, create a default folder.
    
    Raises:
        FileNotFoundError: If zip file doesn't exist
        zipfile.BadZipFile: If file is not a valid zip
        ValueError: If zip is empty or contains no images
    """

    # Validate input file
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Image dataset file not found: {zip_path}. Please check the file path.")

    if not os.path.isfile(zip_path):
        raise ValueError(f"Path is not a file: {zip_path}")

    file_size = os.path.getsize(zip_path)
    if file_size == 0:
        raise ValueError(f"ZIP file is empty: {zip_path}. Please provide a non-empty dataset.")

    # Try to open and validate zip
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Test zip integrity
            bad_file = z.testzip()
            if bad_file is not None:
                raise zipfile.BadZipFile(f"ZIP file is corrupted. Bad file: {bad_file}")
            
            # Check if zip is empty
            if len(z.namelist()) == 0:
                raise ValueError(f"ZIP file is empty: {zip_path}. Please ensure it contains image files.")
            
            # Extract
            z.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid or corrupted ZIP file: {zip_path}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Could not extract ZIP file: {zip_path}. Error: {str(e)}")

    os.makedirs(extract_dir, exist_ok=True)

    # Check extracted structure
    if not os.path.isdir(extract_dir):
        raise ValueError(f"Failed to create extraction directory: {extract_dir}")

    subfolders = [d for d in os.listdir(extract_dir)
                  if os.path.isdir(os.path.join(extract_dir, d))]

    # CASE: flat files (no class subfolders found)
    image_files = [f for f in os.listdir(extract_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    # Validate that we have images
    total_images = 0
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                total_images += 1

    if total_images == 0:
        raise ValueError(f"No image files found in ZIP: {zip_path}. Supported formats: .png, .jpg, .jpeg, .bmp, .gif")

    if len(subfolders) == 0 and len(image_files) > 0:
        class_dir = os.path.join(extract_dir, "class_0")
        os.makedirs(class_dir, exist_ok=True)

        # move images into class_0
        for f in image_files:
            src = os.path.join(extract_dir, f)
            dst = os.path.join(class_dir, f)
            try:
                os.rename(src, dst)
            except Exception as e:
                raise ValueError(f"Failed to organize images: {str(e)}")

    # Final validation: check we have at least some valid images
    valid_count = verify_images(extract_dir)
    if valid_count < 2:
        raise ValueError(f"Too few valid images found ({valid_count}). Minimum 2 images required for training.")

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
