# explainDL/core/config.py
import os

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Main model registry where all trained models are stored
MODEL_REGISTRY_DIR = os.path.join(PROJECT_ROOT, "model_registry")
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

# Default training configuration
TRAINING_CONFIG = {
    "epochs": 12,
    "batch_size": 32,
    "validation_split": 0.15,
    "image_size": (224, 224),
    "text_max_words": 10000,
    "text_max_len": 120,
}
