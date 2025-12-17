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
    
    # Validate inputs
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    model_path = os.path.join(model_dir, "model.h5")
    preproc_path = os.path.join(model_dir, "preprocessor.pkl")
    meta_path = os.path.join(model_dir, "meta.json")

    # ---------------------------------------------------------
    # LOAD MODEL + PREPROCESSOR + METADATA
    # ---------------------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the model was trained successfully.")
    
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preproc_path}. Please ensure the model was trained successfully.")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}. Please ensure the model was trained successfully.")

    try:
        model = load_model(model_path)
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

    # ---------------------------------------------------------
    # TABULAR
    # ---------------------------------------------------------
    if dataset_type == "tabular":
        try:
            df = load_tabular_data(dataset_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading tabular data for prediction: {str(e)}")
        
        try:
            X = preprocessor.transform(df)
        except KeyError as e:
            raise ValueError(f"Feature mismatch: {str(e)}. Prediction dataset must have the same columns as training data.")
        except Exception as e:
            raise ValueError(f"Error preprocessing prediction data: {str(e)}")
        
        filenames = None

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------
    elif dataset_type == "image":
        try:
            extract_dir = os.path.join(model_dir, "predict_images")
            extracted_dir = extract_image_dataset(dataset_path, extract_dir)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading image dataset for prediction: {str(e)}")

        try:
            # preprocess_for_predict returns sorted filenames + normalized array
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
            lines = load_text_file(dataset_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading text file for prediction: {str(e)}")
        
        if not lines or len(lines) == 0:
            raise ValueError("Text file is empty. Please provide text data for prediction.")
        
        # For prediction, text can be unlabelled (one text per line)
        texts = []
        for line in lines:
            # If line contains tab, take text after first tab (for compatibility)
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
    # Use batch prediction for better memory efficiency
    batch_size = 32
    if X.shape[0] > batch_size:
        preds_proba = model.predict(X, batch_size=batch_size, verbose=0)
    else:
        preds_proba = model.predict(X, verbose=0)

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
        report_path = generate_predict_report(pred_class_names, class_names, model_dir,dataset_type)
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
