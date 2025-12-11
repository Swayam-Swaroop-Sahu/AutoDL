"""
config.py
---------
Central configuration file for ExplainDL 2-Mode AutoML System.
Handles:
- Directory paths
- Model registry settings
- Default hyperparameters
- Allowed file formats
- UI configurations
"""

import os

# -------------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# -------------------------------------------------------------------------
# STORAGE & REGISTRY DIRECTORIES
# -------------------------------------------------------------------------
MODEL_REGISTRY_DIR = os.path.join(PROJECT_ROOT, "model_registry")
PREPROCESSOR_REGISTRY_DIR = os.path.join(MODEL_REGISTRY_DIR, "preprocessors")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
os.makedirs(PREPROCESSOR_REGISTRY_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# MODEL & PREPROCESSOR FILE NAMES
# -------------------------------------------------------------------------
MODEL_FILE = os.path.join(MODEL_REGISTRY_DIR, "latest_model.h5")
PREPROCESSOR_FILE = os.path.join(PREPROCESSOR_REGISTRY_DIR, "preprocessor.pkl")
METADATA_FILE = os.path.join(MODEL_REGISTRY_DIR, "metadata.json")


# -------------------------------------------------------------------------
# DEFAULT TRAINING CONFIGURATION
# -------------------------------------------------------------------------
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2

# If True → Auto model selection and tuning enabled by default
AUTO_MODE_DEFAULT = True

# -------------------------------------------------------------------------
# HYPERPARAMETER TUNING (Keras-Tuner)
# -------------------------------------------------------------------------
TUNING_ENABLED = True
TUNING_MAX_TRIALS = 15


# -------------------------------------------------------------------------
# ALLOWED FILE TYPES
# -------------------------------------------------------------------------
TABULAR_FILE_TYPES = [".csv", ".xlsx"]
IMAGE_FILE_TYPE = ".zip"
TEXT_FILE_TYPE = ".txt"

ALL_ALLOWED_FILES = TABULAR_FILE_TYPES + [IMAGE_FILE_TYPE, TEXT_FILE_TYPE]


# -------------------------------------------------------------------------
# UI SETTINGS (for Streamlit)
# -------------------------------------------------------------------------
LIGHT_THEME = {
    "primaryColor": "#4A90E2",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F5F7FA",
    "textColor": "#000000",
    "font": "sans-serif",
}

APP_TITLE = "ExplainDL — Automated Deep Learning Analysis Tool"
APP_DESCRIPTION = (
    "Train deep learning models automatically using your dataset, "
    "interpret results with SHAP/LIME/Grad-CAM, and generate predictions."
)


# -------------------------------------------------------------------------
# PREDICTION SETTINGS
# -------------------------------------------------------------------------
PROBABILITY_THRESHOLD = 0.5  # Default threshold for binary classification

# If True → prediction results will include probability scores
INCLUDE_PROBABILITIES = True


# -------------------------------------------------------------------------
# SAFETY / VALIDATION CONFIGS
# -------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

MIN_ROWS_FOR_TRAINING = 20  # Avoid training on tiny datasets
MIN_IMAGE_COUNT = 10        # Avoid empty ZIP datasets


# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
LOG_FILE = os.path.join(PROJECT_ROOT, "explainDL.log")
ENABLE_LOGGING = True
