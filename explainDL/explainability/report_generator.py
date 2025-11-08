"""
report_generator.py
-------------------
Generates combined explainability reports and saves them as PDF.
Includes confusion matrix and key metrics.
"""

from fpdf import FPDF
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ExplainDLReport:
    def __init__(self, output_path="ExplainDL_Report.pdf"):
        self.output_path = output_path
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(self, title: str):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, title, ln=True, align='C')

    def add_text(self, text: str):
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 8, text)

    def add_image(self, img_path: str, width=170):
        if os.path.exists(img_path):
            self.pdf.image(img_path, w=width)

    def save(self):
        self.pdf.output(self.output_path)
        return self.output_path

def _save_figure_to_png(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _add_confusion_matrix_image(metrics_dict, out_path="confusion_matrix.png"):
    cm = metrics_dict.get("confusion_matrix")
    if cm is None:
        return None
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    _save_figure_to_png(fig, out_path)
    return out_path

def generate_report(metrics_df, metrics_dict=None, shap_fig=None, lime_fig=None, gradcam_fig=None, output_path="ExplainDL_Report.pdf"):
    report = ExplainDLReport(output_path)
    report.add_title("ExplainDL Automated Analysis Report")

    if metrics_df is not None and not metrics_df.empty:
        report.add_text("Performance Summary:\n")
        # Convert metrics_df to string
        report.add_text(metrics_df.to_string(index=False))

    if metrics_dict:
        # Add detailed classification report text if available
        class_report = metrics_dict.get("classification_report")
        if class_report:
            report.add_text("\nClassification Report (per class):")
            try:
                # produce a small DataFrame
                df = pd.DataFrame(class_report).transpose()
                report.add_text(df.to_string())
            except Exception:
                pass

    temp_files = []

    # SHAP figure
    if shap_fig is not None:
        shap_path = "shap_plot.png"
        _save_figure_to_png(shap_fig, shap_path)
        temp_files.append(shap_path)
        report.add_text("\nFeature Importance (SHAP):")
        report.add_image(shap_path)

    # LIME figure
    if lime_fig is not None:
        lime_path = "lime_plot.png"
        _save_figure_to_png(lime_fig, lime_path)
        temp_files.append(lime_path)
        report.add_text("\nLocal Explanation (LIME):")
        report.add_image(lime_path)

    # Grad-CAM
    if gradcam_fig is not None:
        grad_path = "gradcam_plot.png"
        _save_figure_to_png(gradcam_fig, grad_path)
        temp_files.append(grad_path)
        report.add_text("\nGrad-CAM Visualization:")
        report.add_image(grad_path)

    # Confusion matrix
    if metrics_dict:
        cm_path = _add_confusion_matrix_image(metrics_dict, out_path="confusion_matrix.png")
        if cm_path:
            temp_files.append(cm_path)
            report.add_text("\nConfusion Matrix:")
            report.add_image(cm_path)

    path = report.save()

    # cleanup
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

    return path
