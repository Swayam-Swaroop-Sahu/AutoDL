# src/core/pipeline_predict.py

import os
import json
import uuid
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.core.exceptions import AutoDLInputError
from src.core.validation import validate_file_exists, validate_non_empty, validate_prediction_columns
from src.data.tabular_loader import load_tabular_data
from src.data.image_loader import extract_image_dataset
from src.data.text_loader import load_text_file

from src.explainability.report_generator import generate_predict_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_sklearn_model(model) -> bool:
    """Return True if `model` is an sklearn-compatible estimator/Pipeline."""
    if model is None:
        return False
    if isinstance(model, Pipeline):
        return True
    if isinstance(model, BaseEstimator):
        return True
    return False


def predict_pipeline(model_dir: str, dataset_path: str):
    """Runs prediction for a saved model."""
    logger.info("predict_pipeline start — model_dir=%s, dataset_path=%s", model_dir, dataset_path)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. "
            "Why: the trained model may have been deleted or moved. "
            "What to do: train a model first, then run prediction."
        )

    validate_file_exists(dataset_path)

    # BUGFIX Phase 1e item 12: prefer .keras; fall back to .pkl (sklearn) and legacy .h5.
    keras_path = os.path.join(model_dir, "model.keras")
    pkl_path = os.path.join(model_dir, "model.pkl")
    legacy_h5_path = os.path.join(model_dir, "model.h5")
    if os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists(pkl_path):
        model_path = pkl_path
    elif os.path.exists(legacy_h5_path):
        model_path = legacy_h5_path
    else:
        raise FileNotFoundError(
            f"No model file found in {model_dir} (looked for model.keras, model.pkl, model.h5). "
            "Please ensure the model was trained successfully."
        )
    preproc_path = os.path.join(model_dir, "preprocessor.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    # ---------------------------------------------------------
    # LOAD MODEL + PREPROCESSOR + METADATA
    # ---------------------------------------------------------
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(
            f"Preprocessor file not found: '{preproc_path}'. "
            "Why: the model was trained but the preprocessor artifact is missing. "
            "What to do: retrain the model to regenerate all artifacts."
        )

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: '{meta_path}'. "
            "Why: the metadata file is required for prediction configuration. "
            "What to do: retrain the model to regenerate all artifacts."
        )

    # BUGFIX Phase 1e item 12: load .keras or .pkl appropriately.
    try:
        if model_path.endswith(".keras") or model_path.endswith(".h5"):
            model = load_model(model_path)
        else:
            model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Could not load model: {str(e)}. The model file may be corrupted.")

    try:
        preprocessor = joblib.load(preproc_path)
    except Exception as e:
        raise ValueError(f"Could not load preprocessor: {str(e)}. The preprocessor file may be corrupted.")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not load metadata: {str(e)}. The metadata file may be corrupted.")

    dataset_type = meta.get("dataset_type")
    if not dataset_type:
        raise ValueError("Dataset type not found in metadata. Please retrain the model.")

    class_names = meta.get("class_names", None)
    model_name = meta.get("model_name", "Unknown")

    # ---------------------------------------------------------
    # TABULAR
    # ---------------------------------------------------------
    if dataset_type == "tabular":
        try:
            df = load_tabular_data(dataset_path, require_target=False)
        except (FileNotFoundError, ValueError) as e:
            raise AutoDLInputError(
                f"Error loading tabular data for prediction: {str(e)}. "
                "Why: the file could not be parsed. "
                "What to do: check the file format and re-upload."
            )

        validate_non_empty(df, name="prediction dataset")

        # Check column matching
        feature_cols = preprocessor._feature_cols if hasattr(preprocessor, '_feature_cols') else []
        if feature_cols:
            validate_prediction_columns(df, feature_cols)

        try:
            X = preprocessor.transform(df)
        except KeyError as e:
            raise AutoDLInputError(
                f"Feature mismatch: {str(e)}. "
                "Why: the prediction dataset has different columns than the training data. "
                "What to do: ensure prediction data has the same columns as the training data."
            )
        except Exception as e:
            raise AutoDLInputError(
                f"Error preprocessing prediction data: {str(e)}. "
                "Why: the data could not be transformed. "
                "What to do: check data types and values, then re-upload."
            )

        filenames = None

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------
    elif dataset_type == "image":
        try:
            extract_dir = os.path.join(model_dir, "prediction_inputs", uuid.uuid4().hex)
            extracted_dir = extract_image_dataset(dataset_path, extract_dir, require_labels=False)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading image dataset for prediction: {str(e)}")

        try:
            X, filenames = preprocessor.preprocess_for_predict(extracted_dir)
        except Exception as e:
            raise ValueError(f"Error preprocessing images: {str(e)}")

        if X.shape[0] == 0:
            raise ValueError("No valid images found in prediction dataset. Please ensure the ZIP contains valid image files.")

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------
    elif dataset_type == "text":
        try:
            lines = load_text_file(dataset_path, min_lines=1)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading text file for prediction: {str(e)}")

        if not lines or len(lines) == 0:
            raise ValueError("Text file is empty. Please provide text data for prediction.")

        texts = []
        for line in lines:
            if "\t" in line:
                texts.append(line.split("\t", 1)[-1].strip())
            else:
                texts.append(line.strip())

        if not texts or all(not t for t in texts):
            raise ValueError("No valid text found in file. Please ensure the file contains text data.")

        try:
            X = preprocessor.transform(texts)
        except Exception as e:
            raise ValueError(f"Error preprocessing text: {str(e)}")

        filenames = None

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Expected: tabular, image, or text")

    # ---------------------------------------------------------
    # PREDICT (with batching for large datasets)
    # ---------------------------------------------------------
    batch_size = 32
    is_sklearn = _is_sklearn_model(model)

    if is_sklearn:
        # sklearn path: simple .predict() or .predict_proba()
        try:
            preds_proba = model.predict_proba(X)
        except AttributeError:
            # Some classifiers don't have predict_proba (e.g., LinearSVC without calibration)
            # Fall back to predict and create pseudo-probabilities
            preds = model.predict(X)
            # Create one-hot encoded pseudo-probabilities
            n_classes = len(class_names) if class_names else max(preds) + 1
            preds_proba = np.zeros((len(preds), n_classes))
            preds_proba[np.arange(len(preds)), preds] = 1.0
    else:
        # Keras path: supports batch_size and verbose
        if X.shape[0] > batch_size:
            preds_proba = model.predict(X, batch_size=batch_size, verbose=0)
        else:
            preds_proba = model.predict(X, verbose=0)

    # BUGFIX Phase 1c: always argmax; softmax output is shape (N, num_classes).
    # Phase 1d: apply stored binary threshold if available.
    preds = np.argmax(preds_proba, axis=1)

    binary_threshold = meta.get("binary_threshold")
    if binary_threshold is not None and preds_proba.ndim == 2 and preds_proba.shape[1] == 2:
        # Phase 1d: override argmax for binary with stored optimized threshold
        from src.training.threshold import apply_threshold
        preds = apply_threshold(preds_proba, float(binary_threshold), num_classes=2)
        logger.info("Applied stored binary threshold=%.4f", float(binary_threshold))

    # ---------------------------------------------------------
    # MAP PREDICTION INDEX → CLASS NAME
    # ---------------------------------------------------------
    if class_names is not None:
        pred_class_names = [class_names[int(i)] for i in preds]
    else:
        pred_class_names = preds.tolist()

    # ---------------------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------------------
    report_path = None
    try:
        # Pass metadata dict directly instead of relying on relative path resolution
        report_path = generate_predict_report(
            predictions=preds.tolist(),
            class_names=class_names,
            output_dir=model_dir,
            dataset_type=dataset_type,
            model_name=model_name,
            meta=meta,
            prediction_probas=preds_proba if preds_proba.ndim == 2 else None,
        )
    except Exception as e:
        logger.warning("Could not generate prediction report: %s", e)

    # ---------------------------------------------------------
    # RETURN OUTPUT
    # ---------------------------------------------------------
    logger.info("predict_pipeline done — model_dir=%s", model_dir)
    return {
        "predictions": preds.tolist(),
        "prediction_labels": pred_class_names,
        "filenames": filenames,
        "classes": class_names,
        "report_path": report_path,
    }