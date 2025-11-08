"""
tabular_models.py
-----------------
Improved neural network architectures for structured tabular data.
Includes BatchNorm, Dropout and flexible output configuration.
"""

from tensorflow.keras import models, layers, optimizers

def build_mlp(input_shape, output_units=1, activation='sigmoid', loss='binary_crossentropy'):
    """
    Improved MLP for tabular data.
    - BatchNormalization and Dropout added
    - Output units/activation/loss are passed in to support binary/multi-class
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(output_units, activation=activation)
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=loss,
                  metrics=['accuracy'])
    return model
