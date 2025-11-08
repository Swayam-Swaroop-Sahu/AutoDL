"""
shap_explainer.py
-----------------
SHAP-based explainability for tabular models with clearer bar summary.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def explain_with_shap(model, X_sample, max_display=15):
    """
    Generates SHAP feature importance bar plot for tabular models.

    Parameters
    ----------
    model : keras.Model or callable
    X_sample : pandas.DataFrame or numpy array
    max_display : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # If DataFrame given, preserve column names
    if hasattr(X_sample, "values"):
        X_for_shap = X_sample.values
        feature_names = list(X_sample.columns)
    else:
        X_for_shap = X_sample
        feature_names = [f"f{i}" for i in range(X_for_shap.shape[1])]

    # Use a model prediction wrapper for SHAP if needed
    try:
        explainer = shap.Explainer(model.predict, X_for_shap, algorithm="auto")
        shap_values = explainer(X_for_shap)
    except Exception:
        # Fallback to KernelExplainer for unsupported models
        explainer = shap.KernelExplainer(lambda x: model.predict(x), shap.kmeans(X_for_shap, 10))
        shap_values = explainer.shap_values(X_for_shap[:100])

    fig, ax = plt.subplots(figsize=(10, 6))
    # For multiclass shap_values might be list-like; summarise by mean absolute
    try:
        shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
    except Exception:
        # If shap_values is a list (per-class), pick first
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_for_shap, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)

    plt.tight_layout()
    return fig
