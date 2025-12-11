# explainDL/explainability/shap_explainer.py
"""
SHAP explainability helpers for tabular models.

Main function:
    explain_tabular_with_shap(model, X_sample, max_display=15) -> matplotlib.figure.Figure

Notes:
- If shap.Explainer fails for the model, this tries KernelExplainer fallback
  using a small background dataset (kmeans).
- X_sample can be a pandas.DataFrame or numpy array. If DataFrame, feature names are used.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
except Exception as e:
    shap = None  # we'll raise clear error if user tries to call functions requiring shap

import pandas as pd


def explain_tabular_with_shap(model, X_sample, max_display=15):
    """
    Generate a SHAP summary bar plot for tabular input.

    Parameters
    ----------
    model : callable
        A model object with predict method accepting numpy array and returning output.
    X_sample : pandas.DataFrame or numpy.ndarray
        Sample of inputs (ideally small — e.g., 100 rows).
    max_display : int
        Number of features to show.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if shap is None:
        raise ImportError("shap is not installed. Install the shap package to use SHAP explainability.")

    if isinstance(X_sample, pd.DataFrame):
        X_vals = X_sample.values
        feature_names = list(X_sample.columns)
    else:
        X_vals = np.array(X_sample)
        feature_names = [f"f{i}" for i in range(X_vals.shape[1])]

    # Try the recommended fast explainer first
    try:
        explainer = shap.Explainer(model.predict, X_vals, algorithm="auto")
        shap_values = explainer(X_vals)
    except Exception:
        # Fallback to kernel explainer on a small background set
        try:
            # Use kmeans background from shap
            background = shap.kmeans(X_vals, min(10, X_vals.shape[0]))
            explainer = shap.KernelExplainer(model.predict, background.data if hasattr(background, "data") else background)
            shap_values = explainer.shap_values(X_vals[:min(100, X_vals.shape[0])])
        except Exception as e:
            # Give meaningful error
            raise RuntimeError(f"SHAP explanation failed: {e}")

    # Plot summary
    fig = plt.figure(figsize=(8, 6))
    try:
        shap.summary_plot(shap_values, X_vals, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
    except Exception:
        # If shap_values is a list (per-class), try first element
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_vals, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X_vals, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)

    plt.tight_layout()
    return fig
