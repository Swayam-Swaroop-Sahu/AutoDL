"""
image_preprocess.py
-------------------
Handles automatic preprocessing for image datasets.
Resizes, normalizes, augments, and prepares training/validation generators.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image_data(image_dir: str, target_size=(128, 128), batch_size=16, val_split=0.2, seed=42):
    """
    Prepares image data generators for training and validation.
    """

    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")

    # Training augmentation + validation split
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_split
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        directory=image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        directory=image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen
