# src/explainability/__init__.py

from .lime_explainer import explain_tabular_with_lime, explain_text_with_lime
from .gradcam_explainer import generate_gradcam
from .text_explainer import explain_text_sample
from .report_generator import generate_train_report, generate_predict_report
from .importance import compute_permutation_importance
from .narrative import generate_narrative

# SHAP is opt-in only — import explicitly if needed:
#   from src.explainability.shap_explainer import explain_tabular_with_shap

__all__ = [
    "explain_tabular_with_lime",
    "explain_text_with_lime",
    "generate_gradcam",
    "explain_text_sample",
    "generate_train_report",
    "generate_predict_report",
    "compute_permutation_importance",
    "generate_narrative",
]