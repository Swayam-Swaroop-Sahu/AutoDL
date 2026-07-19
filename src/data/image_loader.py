# src/data/image_loader.py

import os
import zipfile
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")


def extract_image_dataset(zip_path: str, extract_dir: str, require_labels: bool = True) -> str:
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
            
            # Reject path-traversal entries before extraction.
            extraction_root = os.path.abspath(extract_dir)
            os.makedirs(extraction_root, exist_ok=True)
            for member in z.infolist():
                target = os.path.abspath(os.path.join(extraction_root, member.filename))
                if os.path.commonpath([extraction_root, target]) != extraction_root:
                    raise ValueError(f"ZIP contains an unsafe path: {member.filename}")
            z.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid or corrupted ZIP file: {zip_path}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Could not extract ZIP file: {zip_path}. Error: {str(e)}")

    os.makedirs(extract_dir, exist_ok=True)

    # Check extracted structure
    if not os.path.isdir(extract_dir):
        raise ValueError(f"Failed to create extraction directory: {extract_dir}")

    # A single top-level folder is a common ZIP convention; use it as the dataset root.
    top_level_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    top_level_images = [f for f in os.listdir(extract_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if len(top_level_dirs) == 1 and not top_level_images:
        nested_root = os.path.join(extract_dir, top_level_dirs[0])
        if any(os.path.isdir(os.path.join(nested_root, d)) for d in os.listdir(nested_root)):
            extract_dir = nested_root

    subfolders = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]

    # CASE: flat files (no class subfolders found)
    image_files = [f for f in os.listdir(extract_dir)
                   if f.lower().endswith(IMAGE_EXTENSIONS)]

    # Validate that we have images
    total_images = 0
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                total_images += 1

    if total_images == 0:
        raise ValueError(f"No image files found in ZIP: {zip_path}. Supported formats: .png, .jpg, .jpeg, .bmp, .gif")

    if require_labels and len(subfolders) < 2:
        raise ValueError("Training images must be arranged in at least two class-name folders.")

    if require_labels:
        class_counts = {
            class_name: sum(
                1
                for root, _, files in os.walk(os.path.join(extract_dir, class_name))
                for f in files
                if f.lower().endswith(IMAGE_EXTENSIONS)
            )
            for class_name in subfolders
        }
        too_small = [name for name, count in class_counts.items() if count < 2]
        if too_small:
            raise ValueError(
                "Each image class needs at least 2 images for training and validation. "
                f"Too small: {too_small}"
            )

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
    min_images = 10 if require_labels else 1
    if valid_count < min_images:
        raise ValueError(f"Too few valid images found ({valid_count}). Minimum {min_images} required.")

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
            if f.lower().endswith(IMAGE_EXTENSIONS):
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
            if f.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, f))

    return paths
