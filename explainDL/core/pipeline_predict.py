# explainDL/core/pipeline_predict.py

import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from explainDL.data.tabular_loader import load_tabular_data
from explainDL.data.image_loader import extract_image_dataset
from explainDL.data.text_loader import load_text_file

from explainDL.preprocessing.tabular_preprocessor import TabularPreprocessor
from explainDL.preprocessing.image_preprocessor import ImagePreprocessor
from explainDL.preprocessing.text_preprocessor import TextPreprocessor

from explainDL.explainability.report_generator import generate_predict_report


def predict_pipeline(model_dir: str, dataset_path: str):
    """Runs prediction for a saved model."""

    model_path = os.path.join(model_dir, "model.h5")
    preproc_path = os.path.join(model_dir, "preprocessor.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    # ---------------------------------------------------------
    # LOAD MODEL + PREPROCESSOR + METADATA
    # ---------------------------------------------------------
    model = load_model(model_path)
    preprocessor = joblib.load(preproc_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dataset_type = meta["dataset_type"]
    class_names = meta.get("class_names", None)

    # ---------------------------------------------------------
    # TABULAR
    # ---------------------------------------------------------
    if dataset_type == "tabular":
        df = load_tabular_data(dataset_path)
        X = preprocessor.transform(df)
        filenames = None

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------
    elif dataset_type == "image":
        extract_dir = os.path.join(model_dir, "predict_images")
        extracted_dir = extract_image_dataset(dataset_path, extract_dir)

        # preprocess_for_predict returns sorted filenames + normalized array
        X, filenames = preprocessor.preprocess_for_predict(extracted_dir)

        if X.shape[0] == 0:
            raise ValueError("No images found in prediction dataset.")

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------
    elif dataset_type == "text":
        lines = load_text_file(dataset_path)
        texts = [line.split("\t", 1)[-1] for line in lines]
        X = preprocessor.transform(texts)
        filenames = None

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    preds_proba = model.predict(X)

    # Binary classification → sigmoid output
    if len(preds_proba.shape) == 2 and preds_proba.shape[1] == 1:
        preds = (preds_proba > 0.5).astype(int).flatten()

    # Multi-class classification → softmax output
    elif len(preds_proba.shape) == 2 and preds_proba.shape[1] > 1:
        preds = np.argmax(preds_proba, axis=1)

    # Regression fallback
    else:
        preds = preds_proba.flatten()

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
        report_path = generate_predict_report(pred_class_names, class_names, model_dir)
    except Exception:
        pass

    # ---------------------------------------------------------
    # RETURN OUTPUT
    # ---------------------------------------------------------
    return {
        "predictions": preds.tolist(),
        "prediction_labels": pred_class_names,
        "filenames": filenames,
        "classes": class_names,
        "report_path": report_path,
    }
