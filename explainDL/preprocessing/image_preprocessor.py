# explainDL/preprocessing/image_preprocessor.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class ImagePreprocessor:
    """
    Handles:
    - Image augmentation for training
    - Validation split
    - Clean image loading for prediction

    Improvements:
    - Stores class_indices after flow_from_directory so caller can save stable mapping.
    - Ensures stable, sorted ordering of filenames for prediction.
    - Converts grayscale images to RGB (3 channels) so models expecting 3-ch input don't crash.
    """

    def __init__(self, target_size=(224, 224)):
        # target_size can be (H, W) or (H, W, C) but we standardize to (H, W)
        if len(target_size) == 3:
            self.target_size = (target_size[0], target_size[1])
        else:
            self.target_size = tuple(target_size)
        self.train_datagen = None
        self.val_datagen = None

        # populated after calling preprocess_for_train
        self.class_indices = None
        self.image_shape = (self.target_size[0], self.target_size[1], 3)  # always 3-channel for models

    # ----------------------------------------------------------------------
    # TRAINING MODE
    # ----------------------------------------------------------------------
    def preprocess_for_train(self, folder_path, batch_size=32, val_split=0.2):
        """
        Returns: (train_gen, val_gen)
        train_gen.class_indices is recorded in self.class_indices
        """

        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=val_split,
            rotation_range=15,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
        )

        train_gen = self.train_datagen.flow_from_directory(
            folder_path,
            target_size=self.target_size,
            class_mode="categorical",
            subset="training",
            batch_size=batch_size,
            shuffle=True,
        )

        val_gen = self.train_datagen.flow_from_directory(
            folder_path,
            target_size=self.target_size,
            class_mode="categorical",
            subset="validation",
            batch_size=batch_size,
            shuffle=False,
        )

        # Save class indices mapping (string -> int)
        # We convert to a dict copy so external modifications won't change internal state.
        self.class_indices = dict(train_gen.class_indices)

        # Provide a stable image shape attribute for model builder
        # Keras ImageDataGenerator yields images shaped (batch, H, W, channels).
        self.image_shape = (self.target_size[0], self.target_size[1], 3)

        return train_gen, val_gen

    # ----------------------------------------------------------------------
    # PREDICTION MODE
    # ----------------------------------------------------------------------
    def preprocess_for_predict(self, folder_path):
        """
        Loads all images from folder_path (and subfolders), returns:
            X: numpy array shape (N, H, W, 3)
            filenames: list[str] stable-sorted in ascending order

        Important:
            - All images are resized to target_size and converted to RGB (3 channels).
            - Output order is stable (sorted path order) so caller can align filenames <-> predictions.
        """

        images = []
        filenames = []

        file_list = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    file_list.append(os.path.join(root, f))

        # Stable ordering is critical so predictions line up reliably
        file_list = sorted(file_list)

        for path in file_list:
            # load_img returns a PIL image; target_size ensures resizing
            img = load_img(path, target_size=self.target_size)  # returns RGB by default when possible
            arr = img_to_array(img)  # shape (H, W, C) — possibly 1 channel for grayscale

            # Ensure 3 channels: if grayscale (C==1), repeat channels
            if arr.ndim == 2:
                # unlikely because img_to_array returns 3-d, but handle defensively
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 1:
                arr = np.concatenate([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                # RGBA -> drop alpha
                arr = arr[..., :3]

            # Normalize to [0,1]
            arr = arr.astype("float32") / 255.0

            images.append(arr)
            filenames.append(os.path.basename(path))

        if len(images) == 0:
            # Return empty array with correct dims if no images found
            return np.zeros((0, self.target_size[0], self.target_size[1], 3), dtype="float32"), []

        return np.stack(images, axis=0), filenames
