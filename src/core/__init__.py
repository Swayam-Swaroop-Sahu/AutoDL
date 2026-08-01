# src/core/__init__.py
#
# Lazy imports: don't eagerly pull in TF / nltk / sklearn.
# Users should import submodules directly:
#   from src.core.pipeline_train import train_pipeline
#   from src.core.validation import validate_target
#   from src.core.exceptions import AutoDLInputError
#
# The package-level names below are provided for convenience but load lazily.

def __getattr__(name):
    if name == "train_pipeline":
        from .pipeline_train import train_pipeline
        return train_pipeline
    if name == "predict_pipeline":
        from .pipeline_predict import predict_pipeline
        return predict_pipeline
    raise AttributeError(f"module 'src.core' has no attribute '{name}'")

__all__ = ["train_pipeline", "predict_pipeline"]