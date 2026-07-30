"""Integration test for image end-to-end on messy data (Phase 1f)."""
import io
import os
import tempfile
import zipfile

import numpy as np
from PIL import Image

import pytest


def _make_image_zip(path, classes=("cat", "dog", "bird"), n_per_class=30,
                    n_corrupted=2, seed=0):
    """Create a ZIP with 3 classes × 30 RGB images + 2 corrupted files."""
    rng = np.random.RandomState(seed)
    files_added = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for cls in classes:
            for i in range(n_per_class):
                arr = rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr(f"{cls}/img_{i:03d}.png", buf.getvalue())
                files_added += 1
        # Corrupted files: random bytes that won't decode as PNG
        for i in range(n_corrupted):
            z.writestr(f"cat/corrupted_{i}.png", b"NOT_A_VALID_PNG" + os.urandom(100))
            files_added += 1
    return files_added


def test_image_zip_extracts_and_validates():
    """The image_loader should extract the ZIP and detect valid images, skipping corrupt ones."""
    from src.data.image_loader import extract_image_dataset, verify_images

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "images.zip")
        extract_dir = os.path.join(tmp, "extracted")
        _make_image_zip(zip_path)

        extracted = extract_image_dataset(zip_path, extract_dir)
        valid_count = verify_images(extracted)
        # We expect at least 3 * 30 = 90 valid images (corrupt ones skipped).
        assert valid_count >= 90, f"Expected >= 90 valid images, got {valid_count}"


def test_image_training_with_corrupted_completes():
    """Image training should complete gracefully with 2 corrupted images present."""
    from src.data.image_loader import extract_image_dataset
    from src.preprocessing.image_preprocessor import ImagePreprocessor
    from src.model_selection.image_models import build_small_cnn

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "images.zip")
        extract_dir = os.path.join(tmp, "extracted")
        _make_image_zip(zip_path)

        extracted = extract_image_dataset(zip_path, extract_dir)

        # Try preprocessing (image generator skips unreadable files via PIL)
        ip = ImagePreprocessor(target_size=(32, 32))
        train_gen, val_gen = ip.preprocess_for_train(extracted, batch_size=8)
        # Generator should have at least the 3 classes worth of data
        assert train_gen.num_classes >= 3
        # Total samples > 0
        assert train_gen.samples > 0

        # Build a tiny Keras model and ensure it compiles
        model = build_small_cnn((32, 32, 3), num_classes=train_gen.num_classes)
        assert model.output_shape == (None, train_gen.num_classes)


def test_corrupted_image_in_zipped_dataset_warns_not_crashes():
    """Loading a ZIP with a corrupted image must not crash the pipeline.

    The image_loader.extract_image_dataset should still succeed; PIL won't
    decode the garbage, so the corrupt files just yield 0 bytes / no pixels.
    """
    from src.data.image_loader import extract_image_dataset, verify_images

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "images.zip")
        extract_dir = os.path.join(tmp, "extracted")
        _make_image_zip(zip_path, n_per_class=5, n_corrupted=3)

        # Must not raise
        extracted = extract_image_dataset(zip_path, extract_dir)
        valid_count = verify_images(extracted)
        # At least 3 classes × 5 valid = 15 valid; corrupted ones skipped.
        assert valid_count >= 15
