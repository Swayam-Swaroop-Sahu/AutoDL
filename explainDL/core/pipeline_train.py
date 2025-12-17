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
# HELPER: text explanations for users
# ----------------------------------------------------------------------
def _write_text_file(target_path: str, lines):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_selection_explanation(model_dir, dataset_type, comparison_data, manual_model_selection):
    """
    Creates a text explanation of how/why the model was selected.
    """
    if comparison_data is None:
        return None

    path = os.path.join(model_dir, "model_selection_explanation.txt")
    lines = []
    lines.append("Model Selection Explanation")
    lines.append("====================================")
    lines.append(f"Dataset Type: {dataset_type}")
    lines.append(f"Selection Mode: {'Manual Override' if manual_model_selection else 'Automatic'}")
    lines.append("")

    selected = comparison_data.get("selected", "N/A")
    reason = comparison_data.get("reason", "Not available")
    lines.append(f"Selected Model: {selected}")
    lines.append(f"Reason: {reason}")
    lines.append("")

    lines.append("Candidate Models:")
    for m in comparison_data.get("models", []):
        lines.append(f"- {m.get('name')}: score={m.get('score'):.4f}")
        desc = m.get("description")
        if desc:
            lines.append(f"  • {desc}")
        pros = m.get("pros")
        cons = m.get("cons")
        if pros:
            lines.append(f"  • Pros: {pros}")
        if cons:
            lines.append(f"  • Cons: {cons}")
    lines.append("")

    lines.append("How to interpret the score:")
    lines.append("- A higher variance score indicates the model produces more differentiated outputs on a quick synthetic probe,")
    lines.append("  which usually correlates with better initial feature separation.")
    lines.append("- This is a heuristic; final performance comes from full training metrics below.")

    _write_text_file(path, lines)
    return path


def _build_training_explanation(model_dir, dataset_type, model_name, metrics, tuning_enabled, comparison_data):
    """
    Creates a text explanation of training outcomes and what they mean.
    """
    path = os.path.join(model_dir, "training_explanation.txt")

    acc = metrics.get("accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1_score")

    lines = []
    lines.append("Training Phase Explanation")
    lines.append("====================================")
    lines.append(f"Dataset Type: {dataset_type}")
    lines.append(f"Model Used: {model_name}")
    if tuning_enabled:
        lines.append("Hyperparameter Tuning: Enabled")
    else:
        lines.append("Hyperparameter Tuning: Disabled")

    # Selection reason summary
    if comparison_data:
        lines.append(f"Model Selection Reason: {comparison_data.get('reason', 'Not available')}")
    lines.append("")

    lines.append("Key Metrics (higher is better for all):")
    if acc is not None:
        lines.append(f"- Accuracy: {acc:.4f}")
    if prec is not None:
        lines.append(f"- Precision (weighted): {prec:.4f}")
    if rec is not None:
        lines.append(f"- Recall (weighted): {rec:.4f}")
    if f1 is not None:
        lines.append(f"- F1-score (weighted): {f1:.4f}")
    lines.append("")

    lines.append("What these metrics mean:")
    lines.append("- Accuracy: Overall fraction of correct predictions.")
    lines.append("- Precision: How often predicted classes are correct (penalizes false positives).")
    lines.append("- Recall: How many true items were recovered (penalizes false negatives).")
    lines.append("- F1-score: Harmonic mean of precision and recall; balanced indicator.")
    lines.append("")

    lines.append("Next steps / interpretation by data type:")
    if dataset_type == "tabular":
        lines.append("- Check feature distributions; consider feature importance (e.g., SHAP) for deeper insight.")
        lines.append("- If recall is low, add more examples of under-represented classes or engineer features.")
    elif dataset_type == "image":
        lines.append("- Review confusion matrix to see which classes are visually similar.")
        lines.append("- Grad-CAM can highlight influential regions; blurry/noisy images may hurt recall.")
    elif dataset_type == "text":
        lines.append("- Inspect misclassified samples for ambiguous phrasing or class overlap.")
        lines.append("- Longer, clearer text often improves precision and recall.")
    else:
        lines.append("- Review class-wise metrics to spot weaknesses.")

    _write_text_file(path, lines)
    return path


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
def train_pipeline(dataset_path: str, enable_tuning: bool = False, tuning_config=None, manual_model_selection: str = None):
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
    selection_explanation_path = None
    training_explanation_path = None

    # =====================================================================
    # TABULAR DATA
    # =====================================================================
    if dataset_type == "tabular":

        df = load_tabular_data(dataset_path)
        preprocessor = TabularPreprocessor()
        X, y = preprocessor.fit_transform(df)

        num_features = X.shape[1]
        num_classes = len(set(y))

        # Generate comparison data first (always)
        _, _, comparison_data = select_best_model("tabular", num_features, num_classes, None)
        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )
        
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
                comparison_data["selected"] = "Tuned-Tabular"
                comparison_data["reason"] = "Hyperparameter tuning was enabled and completed successfully. Model architecture was automatically optimized."

                # save HPs
                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name, comparison_data = select_best_model("tabular", num_features, num_classes, manual_model_selection)

        else:
            model, model_name, comparison_data = select_best_model("tabular", num_features, num_classes, manual_model_selection)

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

        # Generate comparison data first (always)
        _, _, comparison_data = select_best_model("image", input_shape, num_classes, None)
        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )

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
                comparison_data["selected"] = "Tuned-CNN"
                comparison_data["reason"] = "Hyperparameter tuning was enabled and completed successfully. Model architecture was automatically optimized."

                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name, comparison_data = select_best_model("image", input_shape, num_classes, manual_model_selection)

        else:
            model, model_name, comparison_data = select_best_model("image", input_shape, num_classes, manual_model_selection)

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

        # Generate comparison data first (always)
        _, _, comparison_data = select_best_model("text", (vocab_size, max_len), num_classes, None)
        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )

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
                comparison_data["selected"] = "Tuned-Text"
                comparison_data["reason"] = "Hyperparameter tuning was enabled and completed successfully. Model architecture was automatically optimized."

                with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
                    json.dump(make_json_safe(tune_res["best_hyperparameters"]), f, indent=2)

            except Exception:
                model, model_name, comparison_data = select_best_model("text", (vocab_size, max_len), num_classes, manual_model_selection)
        else:
            model, model_name, comparison_data = select_best_model("text", (vocab_size, max_len), num_classes, manual_model_selection)

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

    # Build training explanation after metrics are finalized
    training_explanation_path = _build_training_explanation(
        model_dir,
        dataset_type,
        model_name,
        metrics,
        enable_tuning,
        comparison_data,
    )

    metadata = {
        "model_id": model_id,
        "dataset_type": dataset_type,
        "model_name": model_name,
        "class_names": class_names,
        "metrics": {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")},
        "tuning_enabled": enable_tuning,
        "tuning_config": tuning_config,
        "model_comparison": comparison_data,
        "selection_explanation_path": selection_explanation_path,
        "training_explanation_path": training_explanation_path,
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
        "model_comparison": comparison_data,
        "model_name": model_name,
        "selection_explanation_path": selection_explanation_path,
        "training_explanation_path": training_explanation_path,
    }
