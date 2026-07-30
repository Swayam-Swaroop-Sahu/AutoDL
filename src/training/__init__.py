# src/training/__init__.py

from .trainer import train_model
from .metrics import compute_metrics

__all__ = [
    "train_model",
    "compute_metrics",
]