"""
metrics.py
-----------
Utility functions for evaluating trained models and producing richer metrics.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd

def evaluate_predictions(y_true, y_pred, average='weighted'):
    """
    Computes classification metrics (Accuracy, Precision, Recall, F1, Confusion matrix)
    y_true and y_pred should be 1D arrays of integer labels for binary/multi.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }

    return metrics


def summarize_results(metrics_dict):
    """
    Converts metrics dictionary to a Pandas DataFrame for display.
    """
    df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [
            metrics_dict.get("accuracy", None),
            metrics_dict.get("precision", None),
            metrics_dict.get("recall", None),
            metrics_dict.get("f1_score", None)
        ]
    })
    return df
