# explainDL/utils/random_seed.py
"""
Sets random seed across ALL relevant libraries for reproducibility.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set TF deterministic ops (if supported by installed version)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    return seed
