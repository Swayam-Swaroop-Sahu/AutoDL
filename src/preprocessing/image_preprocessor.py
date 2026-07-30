# src/preprocessing/image_preprocessor.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class ImagePreprocessor:
    """
    Handles:
    - Image augmentation for training
    - Validation split
    - Clean and stable image loading for prediction
    - Automatic binary/categorical class_mode selection
    """

    def __init__(self, target_size=(224, 224)):
        if len(target_size) == 3:
            self.target_size = (target_size[0], target_size[1])
        else:
            self.target_size = tuple(target_size)

        self.train_datagen = None
        self.val_datagen = None

        self.class_indices = None
        self.image_shape = (self.target_size[0], self.target_size[1], 3)

    # ======================================================================
    # TRAINING GENERATORS
    # ======================================================================
    def preprocess_for_train(self, folder_path, batch_size=32, val_split=0.2):
        """
        Chooses class_mode dynamically:
        - 2 classes → 'binary'
        - >2 classes → 'categorical'
        """

        # Init datagen
        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=val_split,
            rotation_range=15,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
        )

        # TEMP generator to detect class count
        temp_gen = self.train_datagen.flow_from_directory(
            folder_path,
            target_size=self.target_size,
            class_mode="categorical",   # always categorical for unified path
            subset="training",
            batch_size=batch_size,
            shuffle=True
        )
        num_classes = temp_gen.num_classes

        # BUGFIX Phase 1c: removed `if num_classes == 2` branch. Always categorical.
        class_mode = "categorical"

        # REAL generators
        train_gen = self.train_datagen.flow_from_directory(
            folder_path,
            target_size=self.target_size,
            class_mode=class_mode,
            subset="training",
            batch_size=batch_size,
            shuffle=True,
        )

        val_gen = self.train_datagen.flow_from_directory(
            folder_path,
            target_size=self.target_size,
            class_mode=class_mode,
            subset="validation",
            batch_size=batch_size,
            shuffle=False,
        )

        self.class_indices = dict(train_gen.class_indices)
        self.image_shape = (self.target_size[0], self.target_size[1], 3)

        return train_gen, val_gen

    # ======================================================================
    # PREDICTION MODE
    # ======================================================================
    def preprocess_for_predict(self, folder_path):
        """
        Returns:
        - X : numpy array of shape (N, H, W, 3)
        - filenames : stable-sorted filenames
        """

        images, filenames = [], []

        file_list = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    file_list.append(os.path.join(root, f))

        file_list = sorted(file_list)

        for path in file_list:
            img = load_img(path, target_size=self.target_size)
            arr = img_to_array(img)

            # Handle grayscale, RGBA
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]

            arr = arr.astype("float32") / 255.0

            images.append(arr)
            filenames.append(os.path.basename(path))

        if not images:
            return np.zeros((0, *self.image_shape)), []

        return np.stack(images), filenames
