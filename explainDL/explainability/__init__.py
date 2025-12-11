# explainDL/explainability/__init__.py

from .shap_explainer import explain_tabular_with_shap
from .lime_explainer import explain_tabular_with_lime, explain_text_with_lime
from .gradcam_explainer import generate_gradcam
from .text_explainer import explain_text_sample
from .report_generator import generate_train_report, generate_predict_report

__all__ = [
    "explain_tabular_with_shap",
    "explain_tabular_with_lime",
    "explain_text_with_lime",
    "generate_gradcam",
    "explain_text_sample",
    "generate_train_report",
    "generate_predict_report",
]
