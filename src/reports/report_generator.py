# src/reports/report_generator.py
"""
Generates PDF reports for ExplainDL using FPDF.
Handles:
- Training Report
- Prediction Report

Supports embedding SHAP/LIME/GradCAM images if provided.
"""

import os
from fpdf import FPDF
from src.utils.file_utils import ensure_dir


# -------------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "ExplainDL Report", ln=True, align="C")
        self.ln(5)


def _add_key_values(pdf, title: str, kv: dict):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", "", 11)

    for k, v in kv.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
    pdf.ln(4)


def _add_image(pdf, image_path: str, label: str):
    if image_path and os.path.exists(image_path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, label, ln=True)
        pdf.image(image_path, w=150)
        pdf.ln(5)


# -------------------------------------------------------------------
# PUBLIC GENERATORS
# -------------------------------------------------------------------
def generate_train_report(model_name: str,
                          dataset_type: str,
                          metrics: dict,
                          model_dir: str,
                          shap_path=None,
                          lime_path=None,
                          gradcam_path=None):
    """
    Creates a PDF summarizing training results.
    Returns saved file path.
    """
    ensure_dir(model_dir)
    save_path = os.path.join(model_dir, "train_report.pdf")

    pdf = PDF()
    pdf.add_page()

    _add_key_values(pdf, "Model Summary", {
        "Model Name": model_name,
        "Dataset Type": dataset_type
    })

    _add_key_values(pdf, "Training Metrics", metrics)

    _add_image(pdf, shap_path, "SHAP Summary")
    _add_image(pdf, lime_path, "LIME Explanation")
    _add_image(pdf, gradcam_path, "Grad-CAM Visualization")

    pdf.output(save_path)
    return save_path


def generate_predict_report(model_name: str,
                            dataset_type: str,
                            predictions,
                            class_names,
                            model_dir: str):
    """
    Prediction report summarizing outputs.
    """
    ensure_dir(model_dir)
    save_path = os.path.join(model_dir, "predict_report.pdf")

    pdf = PDF()
    pdf.add_page()

    _add_key_values(pdf, "Model Summary", {
        "Model Name": model_name,
        "Dataset Type": dataset_type,
        "Total Predictions": len(predictions)
    })

    # Class distribution
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Class Distribution", ln=True)

    pdf.set_font("Arial", "", 11)
    from collections import Counter
    counts = Counter(predictions)

    for c, cnt in counts.items():
        name = class_names[c] if class_names else str(c)
        pdf.multi_cell(0, 8, f"{name}: {cnt}")

    pdf.output(save_path)
    return save_path
