# src/model_selection/image_models.py
"""
Candidate image models.
We provide:
- Small custom CNN
- MobileNetV2 top (transfer learning)
- EfficientNetB0 top (transfer learning)

Important: transfer-learning backbones set `trainable=False` by default for quick runs.
"""

from tensorflow.keras import layers, models, optimizers, applications


def _compile_top(base_model: models.Model, num_classes: int):
    """
    Adds a classification head safely for Functional AND Sequential models.
    """
    # CASE: Sequential
    if isinstance(base_model, models.Sequential):
        if num_classes == 2:
            base_model.add(layers.Dense(1, activation="sigmoid"))
            base_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            base_model.add(layers.Dense(num_classes, activation="softmax"))
            base_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return base_model

    # CASE: Functional → append head correctly
    x = base_model.output
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model



def build_small_cnn(input_shape, num_classes: int):
    """
    A small CNN from scratch (good for small datasets).
    input_shape: (H, W, C)
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    model = models.Model(inputs=inputs, outputs=x)
    return _compile_top(model, num_classes)


def build_mobilenet(input_shape, num_classes: int):
    """
    MobileNetV2 backbone (transfer learning).
    input_shape must have 3 channels and reasonable spatial dims.
    """
    # include_top=False so we can append our head
    base = applications.MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    model = models.Model(inputs=base.input, outputs=x)
    return _compile_top(model, num_classes)


def build_efficientnet(input_shape, num_classes: int):
    """
    EfficientNetB0 backbone.
    """
    base = applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    model = models.Model(inputs=base.input, outputs=x)
    return _compile_top(model, num_classes)
