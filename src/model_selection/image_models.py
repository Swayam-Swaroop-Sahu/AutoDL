# src/model_selection/image_models.py
"""
Candidate image models (Phase 1c — unified multiclass).

All image models use a single code path:
  - final Dense(units=num_classes, activation="softmax")
  - loss="sparse_categorical_crossentropy"
  - Integer labels decoded via argmax(predict_proba, axis=1)
  - Even binary → softmax over 2 classes (no separate sigmoid branch)

No binary-vs-multiclass branches remain.
"""

from tensorflow.keras import layers, models, optimizers, applications


def _compile_top(base_model: models.Model, num_classes: int):
    """
    Adds a unified classification head and compiles.
    Handles both Sequential and Functional (transfer-learning) base models.

    BUGFIX Phase 1c: removed `if num_classes == 2` branch.
    """
    # BUGFIX Phase 1c: always use num_classes softmax output.
    loss = "sparse_categorical_crossentropy"

    if isinstance(base_model, models.Sequential):
        base_model.add(layers.Dense(num_classes, activation="softmax"))
        base_model.compile(
            optimizer="adam", loss=loss, metrics=["accuracy"],
        )
        return base_model

    # Functional → append head correctly
    x = base_model.output
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_small_cnn(input_shape, num_classes: int):
    """A small CNN from scratch (good for small datasets). input_shape: (H, W, C)."""
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
    """MobileNetV2 backbone (transfer learning)."""
    base = applications.MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    model = models.Model(inputs=base.input, outputs=x)
    return _compile_top(model, num_classes)


def build_efficientnet(input_shape, num_classes: int):
    """EfficientNetB0 backbone (transfer learning)."""
    base = applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    model = models.Model(inputs=base.input, outputs=x)
    return _compile_top(model, num_classes)
