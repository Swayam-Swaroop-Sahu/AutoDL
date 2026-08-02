# src/core/config.py
"""
Central configuration for AutoDL.
All constants live here — single source of truth.
"""

import os
import random
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# -------------------------------------------------------------------------
# STORAGE & REGISTRY DIRECTORIES
# -------------------------------------------------------------------------
MODEL_REGISTRY_DIR = os.path.join(PROJECT_ROOT, "model_registry")
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# DEFAULT TRAINING CONFIGURATION
# -------------------------------------------------------------------------
TRAINING_CONFIG = {
    "epochs": 12,
    "batch_size": 32,
    "validation_split": 0.15,
    "image_size": (224, 224),
    "text_max_words": 10000,
    "text_max_len": 120,
}

# -------------------------------------------------------------------------
# RANDOM SEED — global reproducibility
# -------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED):
    """Sets PYTHONHASHSEED, random, numpy, and TensorFlow random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return seed


# -------------------------------------------------------------------------
# BINARY THRESHOLD
# -------------------------------------------------------------------------
DEFAULT_BINARY_THRESHOLD = 0.5

# -------------------------------------------------------------------------
# SUCCESSIVE-HALVING SEARCH CONFIG
# -------------------------------------------------------------------------
SEARCH_CV_FOLDS = 5
SEARCH_BUDGET_FRACTION = 0.5
SEARCH_MIN_RESOURCE = 100
SEARCH_MAX_CANDIDATES = 6

# -------------------------------------------------------------------------
# EXPLAINABILITY
# -------------------------------------------------------------------------
# SHAP is opt-in only — set to True to enable SHAP explainability in reports.
USE_SHAP: bool = False

# -------------------------------------------------------------------------
# STAGE TIMEOUTS (circuit breaker)
# -------------------------------------------------------------------------
STAGE_TIMEOUTS = {
    "load": 120,
    "preprocess": 180,
    "target_detect": 60,
    "quality": 30,
    "search": 300,
    "train": 600,
    "evaluate": 120,
    "report": 60,
    "save": 30,
}