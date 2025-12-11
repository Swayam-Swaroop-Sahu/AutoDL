# explainDL/core/pipeline_train.py

import os
import json
import uuid
import joblib
import numpy as np

from explainDL.core.config import MODEL_REGISTRY_DIR
from explainDL.data.detect_type import detect_dataset_type
from explainDL.data.tabular_loader import load_tabular_data
from explainDL.data.image_loader import extract_image_dataset
from explainDL.data.text_loader import load_text_file, parse_labelled_text

from explainDL.preprocessing.tabular_preprocessor import TabularPreprocessor
from explainDL.preprocessing.image_preprocessor import ImagePreprocessor
from explainDL.preprocessing.text_preprocessor import TextPreprocessor

from explainDL.model_selection.selector import select_best_model

# tuner imported best-effort
try:
    from explainDL.model_selection.tuner import tune_model
except Exception:
    tune_model = None

from explainDL.training.trainer import train_model
from explainDL.training.metrics import compute_metrics
from explainDL.explainability.report_generator import generate_train_report


# ----------------------------------------------------------------------
# JSON SAFE CONVERTER
# ----------------------------------------------------------------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ----------------------------------------------------------------------
# MAIN TRAIN PIPELINE
# ----------------------------------------------------------------------
def train_pipeline(dataset_path: str, enable_tuning: bool = False, tuning_config=None):
    """
    Full training pipeline including:
    - preprocessing
    - (optional) hyperparameter tuning
    - training
    - evaluation
    - explainability report

    tuning_config example:
        {
            "max_trials": 6,
            "epochs": 8,
            "tune_learning_rate": True,
            "tune_hidden_units": True,
            "tune_dropout": False
        }
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_type = detect_dataset_type(dataset_path)

    # Create model directory
    model_id = str(uuid.uuid4())[:8]
    model_dir = os.path.join(MODEL_REGISTRY_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    preprocessor = None
    model = None
    model_name = None
    class_names = None
    metrics = {}
    history = {}

    # =====================================================================
    # TABULAR DATA
    # =====================================================================
    if dataset_type == "tabular":

        df = load_tabular_data(dataset_path)
        preprocessor = TabularPreprocessor()
        X, y = preprocessor.fit_transform(df)

        num_features = X.shape[1]
        num_classes = len(set(y))

        # --------------------
        # HYPERPARAMETER TUNING
        # --------------------
        if enable_tuning and tune_model is not None and tuning_config is not None:
            try:
                tune_res = tune_model(
                    model_type="tabular",
                    train_data=(X, y),
                    input_shape=num_features,
                    num_classes=num_classes,
                    max_trials=tuning_config["max_trials"],
                    epochs=tuning_config["epochs"],
                    directory=os.path.join(model_dir, "tuning_logs"),
                    project_name="tab_tune",
                )

                model = tune_res["best_model"]
                model_name = "Tuned-Tabular"

                # save HPs
                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name = select_best_model("tabular", num_features, num_classes)

        else:
            model, model_name = select_best_model("tabular", num_features, num_classes)

        # TRAIN
        history, _ = train_model(model, "tabular", (X, y))

        # METRICS
        y_proba = model.predict(X)
        y_pred = np.argmax(y_proba, axis=1) if y_proba.ndim == 2 else (y_proba > 0.5).astype(int).flatten()

        metrics = compute_metrics(np.array(y).astype(int), y_pred)
        metrics["y_true"] = list(map(int, y))
        metrics["y_pred"] = y_pred.tolist()

        class_names = sorted(list(set(int(v) for v in y)))

    # =====================================================================
    # IMAGE DATA
    # =====================================================================
    elif dataset_type == "image":

        extract_dir = os.path.join(model_dir, "images_extracted")
        extracted = extract_image_dataset(dataset_path, extract_dir)

        preprocessor = ImagePreprocessor()
        train_gen, val_gen = preprocessor.preprocess_for_train(extracted)

        num_classes = train_gen.num_classes
        input_shape = train_gen.image_shape

        if enable_tuning and tune_model is not None and tuning_config is not None:
            try:
                tune_res = tune_model(
                    model_type="image",
                    train_data=(train_gen, val_gen),
                    input_shape=input_shape,
                    num_classes=num_classes,
                    max_trials=tuning_config["max_trials"],
                    epochs=tuning_config["epochs"],
                    directory=os.path.join(model_dir, "tuning_logs"),
                    project_name="image_tune",
                )

                model = tune_res["best_model"]
                model_name = "Tuned-CNN"

                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name = select_best_model("image", input_shape, num_classes)

        else:
            model, model_name = select_best_model("image", input_shape, num_classes)

        history, _ = train_model(model, "image", (train_gen, val_gen))

        y_true = np.array(val_gen.classes)
        y_proba = model.predict(val_gen)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = compute_metrics(y_true, y_pred)
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_pred.tolist()

        class_names = [name for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]

    # =====================================================================
    # TEXT DATA
    # =====================================================================
    elif dataset_type == "text":

        lines = load_text_file(dataset_path)
        texts, labels = parse_labelled_text(lines)

        preprocessor = TextPreprocessor()
        X, y = preprocessor.fit_transform(texts, labels)

        vocab_size = preprocessor.max_words
        max_len = preprocessor.max_len
        num_classes = len(set(labels))

        if enable_tuning and tune_model is not None and tuning_config is not None:
            try:
                tune_res = tune_model(
                    model_type="text",
                    train_data=(X, y),
                    input_shape=(vocab_size, max_len),
                    num_classes=num_classes,
                    max_trials=tuning_config["max_trials"],
                    epochs=tuning_config["epochs"],
                    directory=os.path.join(model_dir, "tuning_logs"),
                    project_name="text_tune",
                )

                model = tune_res["best_model"]
                model_name = "Tuned-Text"

                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name = select_best_model("text", (vocab_size, max_len), num_classes)
        else:
            model, model_name = select_best_model("text", (vocab_size, max_len), num_classes)

        history, _ = train_model(model, "text", (X, y))

        y_proba = model.predict(X)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = compute_metrics(np.array(y).astype(int), y_pred)
        metrics["y_true"] = list(map(int, y))
        metrics["y_pred"] = y_pred.tolist()

        class_names = sorted(list(set(labels)))

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # ----------------------------------------------------------------------
    # SAVE ARTIFACTS
    # ----------------------------------------------------------------------
    model_path = os.path.join(model_dir, "model.h5")
    model.save(model_path)

    preproc_path = os.path.join(model_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preproc_path)

    metadata = {
        "model_id": model_id,
        "dataset_type": dataset_type,
        "model_name": model_name,
        "class_names": class_names,
        "metrics": {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")},
        "tuning_enabled": enable_tuning,
        "tuning_config": tuning_config,
    }

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(make_json_safe(metadata), f, indent=2)

    # ----------------------------------------------------------------------
    # GENERATE TRAIN REPORT (loss, acc, confusion, explainability)
    # ----------------------------------------------------------------------
    try:
        report_path = generate_train_report(history, metrics, model_name, model_dir)
    except Exception:
        report_path = None

    # RETURN
    return {
        "model_id": model_id,
        "model_dir": model_dir,
        "model_path": model_path,
        "preprocessor_path": preproc_path,
        "report_path": report_path,
        "dataset_type": dataset_type,
        "class_names": class_names,
    }
