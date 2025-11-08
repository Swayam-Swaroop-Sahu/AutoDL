"""
image_models.py
---------------
Defines CNN architectures for image data.
Includes EfficientNet fallback to a small CNN if necessary.
"""

from tensorflow.keras import layers, models, applications, optimizers

def build_cnn(input_shape=(128, 128, 3), output_units=1, activation='sigmoid', loss='binary_crossentropy'):
    """
    Builds a CNN model for image classification.

    Parameters
    ----------
    input_shape : tuple
        Image input shape, e.g. (128,128,3)
    output_units : int
        Number of output neurons (1 for binary, N for multi-class)
    activation : str
        Activation for final layer ('sigmoid' or 'softmax')
    loss : str
        Loss function name

    Returns
    -------
    model : keras.Model
    """
    try:
        # Try EfficientNetB0 transfer learning
        base_model = applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')

        # Freeze early layers, fine-tune last 30%
        fine_tune_at = int(len(base_model.layers) * 0.7)
        for i, layer in enumerate(base_model.layers):
            layer.trainable = False if i < fine_tune_at else True

        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(output_units, activation=activation)(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])

        print("[ExplainDL] Using EfficientNetB0 backbone for image model.")
        return model

    except Exception as e:
        # Fallback small CNN (works offline)
        print(f"[ExplainDL] EfficientNet unavailable or failed ({e}). Using fallback CNN.")

        if output_units == 1:
            final_activation = 'sigmoid'
            final_loss = 'binary_crossentropy'
        else:
            final_activation = 'softmax'
            final_loss = 'categorical_crossentropy'

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(output_units, activation=final_activation)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=final_loss, metrics=['accuracy'])
        return model
