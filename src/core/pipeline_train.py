# src/core/pipeline_train.py

import os
import json
import uuid
import joblib
import numpy as np

from src.core.config import MODEL_REGISTRY_DIR, RANDOM_SEED, set_global_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)
from src.data.detect_type import detect_dataset_type
from src.data.tabular_loader import load_tabular_data
from src.data.image_loader import extract_image_dataset
from src.data.text_loader import load_text_file, parse_labelled_text

from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.text_preprocessor import TextPreprocessor

from src.training.trainer import train_model
from src.explainability.report_generator import generate_train_report
from src.explainability.report_generator import generate_train_report


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
            lines.append(f"  - {desc}")
        pros = m.get("pros")
        cons = m.get("cons")
        if pros:
            lines.append(f"  - Pros: {pros}")
        if cons:
            lines.append(f"  - Cons: {cons}")
    lines.append("")

    lines.append("How to interpret the score:")
    lines.append("- A score of 1 marks the scale-based baseline recommendation; 0 marks other available architectures.")
    lines.append("- It is not an accuracy measurement. Use the held-out validation metrics to judge model quality.")

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
def train_pipeline(dataset_path: str, enable_tuning: bool = False, tuning_config=None, manual_model_selection: str = None, target_col: str = None):
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
    set_global_seed(RANDOM_SEED)
    logger.info("train_pipeline start — dataset=%s, tuning=%s", dataset_path, enable_tuning)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Detect dataset type with error handling
    try:
        dataset_type = detect_dataset_type(dataset_path)
    except Exception as e:
        raise ValueError(f"Could not detect dataset type: {str(e)}. Please ensure the file is in a supported format (.csv, .xlsx, .txt, or .zip).")

    if dataset_type == "unknown":
        raise ValueError(
            f"Could not determine dataset type from file: {dataset_path}. "
            "Supported formats:\n"
            "- Tabular: .csv, .xlsx\n"
            "- Image: .zip (containing image files in class folders)\n"
            "- Text: .txt (format: label<TAB>text or label,text per line)"
        )

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
    search_results = None

    # =====================================================================
    # TABULAR DATA
    # =====================================================================
    if dataset_type == "tabular":
        try:
            df = load_tabular_data(dataset_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except ValueError as e:
            raise ValueError(f"Tabular data loading error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading tabular data: {str(e)}. Please check the file format and content.")

        # --- Target detection ---
        from src.target_detection import resolve_target
        resolved, status = resolve_target(df, target_col=target_col)
        logger.info("target detection: resolved=%s, status=%s", resolved, status)

        if status == "human_required":
            raise ValueError(
                f"Target column is ambiguous. Ranked candidates: {resolved}. "
                f"Please pass an explicit target_col= to disambiguate."
            )
        if status == "not_classification":
            raise ValueError(
                "No classification target detected in this dataset. "
                "Every column scored too low to be a class label. "
                "If this is a classification dataset, pass an explicit target_col=."
            )

        detected_target = str(resolved)

        try:
            preprocessor = TabularPreprocessor()
            X, y = preprocessor.fit_transform(df, target_col=detected_target)
        except ValueError as e:
            raise ValueError(f"Tabular preprocessing error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error during preprocessing: {str(e)}")

        num_features = X.shape[1]
        num_classes = len(set(y))
        n_samples = len(X)

        # --- Model search via successive halving ---
        from src.model_selection.search import (
            successive_halving_search, DEFAULT_SCORING,
        )
        from src.model_selection.tabular_candidates import get_tabular_candidates
        candidates = get_tabular_candidates(n_samples)
        best_cfg, best_score, all_results = successive_halving_search(
            candidates, X.to_numpy() if hasattr(X, "to_numpy") else X,
            y, time_budget_sec=600, scoring=DEFAULT_SCORING,
        )
        search_results = all_results

        if not best_cfg or not best_cfg.get("factory"):
            # Fallback: pick the first candidate
            logger.warning("Search returned no winner; falling back to first candidate")
            best_cfg = {"name": candidates[0].name, "factory": candidates[0]}
            best_score = -1.0

        logger.info("Search winner: %s (score=%.4f)", best_cfg["name"], best_score)
        model_name = best_cfg["name"]
        search_scoring = "balanced_accuracy"
        try:
            model = best_cfg["factory"].build()
        except Exception:
            model = best_cfg["factory"].factory()

        # Build comparison data for reports
        comparison_data = {
            "selected": model_name,
            "reason": (
                f"'{model_name}' won successive-halving search with "
                f"{search_scoring}={best_score:.4f}."
                if best_score >= 0 else
                f"'{model_name}' selected as fallback (search did not produce valid scores)."
            ),
            "models": [
                {
                    "name": r.get("name", "?"),
                    "score": r.get("score", -1.0),
                    "description": r.get("description", ""),
                    "params": r.get("params", ""),
                    "pros": r.get("pros", ""),
                    "cons": r.get("cons", ""),
                }
                for r in all_results
            ],
        }

        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )

        # TRAIN (fit the winner on full data again inside train_model)
        history, metrics = train_model(model, "tabular", (X, y))
        class_names = preprocessor.target_encoder.classes_.tolist()

    # =====================================================================
    # IMAGE DATA
    # =====================================================================
    elif dataset_type == "image":
        try:
            extract_dir = os.path.join(model_dir, "images_extracted")
            extracted = extract_image_dataset(dataset_path, extract_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except ValueError as e:
            raise ValueError(f"Image dataset error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading image dataset: {str(e)}. Please ensure the ZIP file contains valid images in class folders.")

        preprocessor = ImagePreprocessor()
        train_gen, val_gen = preprocessor.preprocess_for_train(extracted)

        num_classes = train_gen.num_classes
        input_shape = train_gen.image_shape
        n_samples = train_gen.samples

        # --- Model search ---
        from src.model_selection.image_candidates import get_image_candidates
        candidates = get_image_candidates(n_samples)
        # Image search uses a dummy X, y for the search — the search for images
        # is still pending proper integration. For now, use first candidate.
        best_cfg, best_score, all_results = {"name": candidates[0].name, "factory": candidates[0]}, -1.0, []
        model_name = candidates[0].name
        model = candidates[0].build()

        # Build comparison data
        comparison_data = {
            "selected": model_name,
            "reason": f"'{model_name}' selected based on dataset size tier ({n_samples} samples).",
            "models": [
                {
                    "name": c.name,
                    "score": 1.0 if i == 0 else 0.0,
                    "description": c.description,
                    "params": c.params,
                    "pros": c.pros,
                    "cons": c.cons,
                }
                for i, c in enumerate(candidates)
            ],
        }

        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )

        history, metrics = train_model(model, "image", (train_gen, val_gen))

        class_names = [name for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]

    # =====================================================================
    # TEXT DATA
    # =====================================================================
    elif dataset_type == "text":
        try:
            lines = load_text_file(dataset_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except ValueError as e:
            raise ValueError(f"Text file loading error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading text file: {str(e)}")

        try:
            texts, labels = parse_labelled_text(lines)
        except ValueError as e:
            raise ValueError(f"Text parsing error: {str(e)}. Expected format: 'label<TAB>text' or 'label,text' per line.")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing text data: {str(e)}")

        if texts is None or labels is None:
            raise ValueError(
                "Text file does not appear to be labelled. "
                "For training, please use format: 'label<TAB>text' or 'label,text' per line. "
                "For prediction, use unlabelled text (one text per line)."
            )

        try:
            preprocessor = TextPreprocessor()
            X, y = preprocessor.fit_transform(texts, labels)
        except ValueError as e:
            raise ValueError(f"Text preprocessing error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error during text preprocessing: {str(e)}")

        vocab_size = preprocessor.max_words
        max_len = preprocessor.max_len
        num_classes = len(set(labels))
        n_samples = len(X)

        # --- Model search ---
        from src.model_selection.search import successive_halving_search
        from src.model_selection.text_candidates import get_text_candidates
        candidates = get_text_candidates(n_samples)
        best_cfg, best_score, all_results = successive_halving_search(
            candidates, X, y, time_budget_sec=600, scoring="balanced_accuracy",
        )
        search_results = all_results

        if not best_cfg or not best_cfg.get("factory"):
            logger.warning("No winner from text search; falling back to first candidate")
            best_cfg = {"name": candidates[0].name, "factory": candidates[0]}
            best_score = -1.0

        logger.info("Search: %s (score=%.4f)", best_cfg["name"], best_score)
        model_name = best_cfg["name"]
        try:
            model = best_cfg["factory"].build()
        except Exception:
            model = best_cfg["factory"].factory()

        # Build comparison data
        comparison_data = {
            "selected": model_name,
            "reason": (
                f"'{model_name}' won successive-halving search with "
                f"balanced_accuracy={best_score:.4f}."
                if best_score >= 0 else
                f"'{model_name}' selected as fallback."
            ),
            "models": [
                {
                    "name": r.get("name", "?"),
                    "score": r.get("score", -1.0),
                    "description": r.get("description", ""),
                    "params": r.get("params", ""),
                    "pros": r.get("pros", ""),
                    "cons": r.get("cons", ""),
                }
                for r in all_results
            ],
        }

        selection_explanation_path = _build_selection_explanation(
            model_dir, dataset_type, comparison_data, manual_model_selection
        )

        history, metrics = train_model(model, "text", (X, y))
        class_names = preprocessor.label_encoder.classes_.tolist()

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
        False,  # tuning no longer supported in v2
        comparison_data,
    )

    metadata = {
        "model_id": model_id,
        "dataset_type": dataset_type,
        "model_name": model_name,
        "class_names": class_names,
        "metrics": {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")},
        "tuning_enabled": False,
        "model_comparison": comparison_data,
        "search_results": search_results,
        "selection_explanation_path": selection_explanation_path,
        "training_explanation_path": training_explanation_path,
    }

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(make_json_safe(metadata), f, indent=2)

    # ----------------------------------------------------------------------
    # GENERATE ENHANCED TRAIN REPORT (loss, acc, confusion, explainability, model selection)
    # ----------------------------------------------------------------------
    try:
        report_path = generate_train_report(
            history, 
            metrics, 
            model_name, 
            model_dir,
            model_comparison=comparison_data,
            selection_explanation_path=selection_explanation_path,
            training_explanation_path=training_explanation_path
        )
    except Exception as e:
        # Log error but don't fail the pipeline
        import traceback
        logger.warning("Could not generate enhanced PDF report: %s", e)
        logger.debug(traceback.format_exc())
        report_path = None

    logger.info("train_pipeline done — model_id=%s, model_name=%s", model_id, model_name)

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
