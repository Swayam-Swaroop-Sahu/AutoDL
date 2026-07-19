# src/training/callbacks.py

"""
Provides reusable callback sets.
Many modules do NOT use this yet because trainer.py
already defines a good callback set.
"""

import os
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


def get_callbacks(save_dir="checkpoints", monitor="val_loss"):
    os.makedirs(save_dir, exist_ok=True)

    return [
        EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.keras"),
            monitor=monitor,
            save_best_only=True
        )
    ]
