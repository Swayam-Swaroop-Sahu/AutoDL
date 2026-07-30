"""Preprocessing public API with lazy modality imports.

Importing a tabular helper must not require TensorFlow or image dependencies.
"""

__all__ = ["TabularPreprocessor", "ImagePreprocessor", "TextPreprocessor"]


def __getattr__(name):
    if name == "TabularPreprocessor":
        from .tabular_preprocessor import TabularPreprocessor
        return TabularPreprocessor
    if name == "ImagePreprocessor":
        from .image_preprocessor import ImagePreprocessor
        return ImagePreprocessor
    if name == "TextPreprocessor":
        from .text_preprocessor import TextPreprocessor
        return TextPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")