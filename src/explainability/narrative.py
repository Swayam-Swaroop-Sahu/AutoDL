"""Plain-English narrative summary of a training run.

Generates a 2–3 sentence human-readable description suitable for embedding
in reports or meta.json.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def generate_narrative(meta: Dict, quality: Optional[Dict] = None) -> str:
    """Build a 2–3 sentence plain-English summary of the training run.

    Parameters
    ----------
    meta : dict
        Must include keys: ``model_name``, ``metrics`` (with ``accuracy``).
        May optionally include ``feature_importance`` (list of dicts from
        :func:`compute_permutation_importance`).
    quality : dict or None
        Quality summary as returned by :func:`src.quality.summarize.summarize_quality`.
        Keys: ``warnings`` (list), ``passed`` (bool).

    Returns
    -------
    str
        2–3 sentence narrative string.
    """
    model_name = meta.get("model_name", "the model")
    metrics = meta.get("metrics", {})
    accuracy = metrics.get("accuracy")

    # --- Sentence 1: What model was used and how accurate is it ---
    if accuracy is not None:
        sentence1 = (
            f"The model ({model_name}) achieves {accuracy:.0%} accuracy "
            f"on the held-out validation set."
        )
    else:
        sentence1 = (
            f"The model ({model_name}) was trained successfully."
        )

    # --- Sentence 2: What drives the predictions ---
    feature_importance = meta.get("feature_importance")
    if feature_importance and len(feature_importance) > 0:
        top = feature_importance[0]
        sentence2 = (
            f"It relies most on '{top['feature']}' to make its predictions."
        )
    else:
        sentence2 = (
            "The most influential features were not computed for this run."
        )

    # --- Sentence 3: Any quality warnings ---
    warning_text = ""
    if quality and not quality.get("passed", True):
        warnings = quality.get("warnings", [])
        if warnings:
            issue_types = list({w.get("issue", "issue") for w in warnings})
            if len(issue_types) == 1:
                warning_text = (
                    f"Note: the training data has a potential {issue_types[0].replace('_', ' ')} issue "
                    f"that may affect reliability."
                )
            else:
                friendly = [t.replace("_", " ") for t in issue_types]
                warning_text = (
                    f"Note: the training data has several quality issues "
                    f"({', '.join(friendly)}) that may affect reliability."
                )
        else:
            warning_text = (
                "Note: some data quality checks did not pass, which may affect reliability."
            )
    else:
        warning_text = (
            "No major data quality issues were detected."
        )

    parts = [p for p in [sentence1, sentence2, warning_text] if p]
    return " ".join(parts)