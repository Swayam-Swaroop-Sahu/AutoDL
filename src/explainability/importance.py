"""Permutation feature importance for tabular models.

Uses sklearn.inspection.permutation_importance to compute how much each
feature contributes to model performance.  Falls back gracefully when the
model type isn't supported (e.g. some Keras models).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_repeats: int = 5,
    scoring: str = "accuracy",
    feature_names: Optional[List[str]] = None,
) -> Optional[List[Dict[str, float]]]:
    """Compute permutation feature importance.

    Parameters
    ----------
    model : trained model (must have a ``.predict()``-like method).
    X_val : feature matrix (numpy array or array-like).
    y_val : true labels (numpy array).
    n_repeats : number of times to permute each feature (default 5).
    scoring : scoring metric for sklearn.permutation_importance (default "accuracy").
    feature_names : optional feature name labels. If omitted, generic names are used.

    Returns
    -------
    list of dict
        Each entry: {"feature": str, "importance": float, "std": float}
        Sorted descending by importance.  Returns ``None`` if the model type
        does not support permutation importance or an error occurs.
    """
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        logger.warning(
            "sklearn.inspection.permutation_importance not available; "
            "skipping feature importance"
        )
        return None

    # ------------------------------------------------------------------
    # Guard: model must have a predict method
    # ------------------------------------------------------------------
    if not hasattr(model, "predict"):
        logger.warning(
            "Model %s does not have a .predict() method; "
            "skipping permutation importance.",
            type(model).__name__,
        )
        return None

    # ------------------------------------------------------------------
    # Guard: X_val and y_val shouldn't be empty
    # ------------------------------------------------------------------
    X_arr = np.asarray(X_val)
    y_arr = np.asarray(y_val)
    if X_arr.shape[0] == 0 or y_arr.shape[0] == 0:
        logger.warning(
            "Empty validation arrays passed to permutation importance; skipping."
        )
        return None

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    try:
        result = permutation_importance(
            model,
            X_arr,
            y_arr,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
        )
    except Exception as exc:
        logger.warning(
            "Permutation importance failed for model %s: %s. "
            "Feature importance not available for this model type.",
            type(model).__name__,
            exc,
        )
        return None

    # ------------------------------------------------------------------
    # Build sorted list
    # ------------------------------------------------------------------
    n_features = X_arr.shape[1]
    if feature_names and len(feature_names) == n_features:
        names = feature_names
    else:
        names = [f"feature_{i}" for i in range(n_features)]

    ranked = sorted(
        [
            {
                "feature": names[i],
                "importance": float(result.importances_mean[i]),
                "std": float(result.importances_std[i]),
            }
            for i in range(n_features)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    logger.info(
        "Permutation importance computed — top feature: '%s' (importance=%.4f)",
        ranked[0]["feature"],
        ranked[0]["importance"],
    )
    return ranked