"""
ExplainDL – Extended Report Generator
-------------------------------------

Adds:
- Confusion matrix
- Precision / Recall / F1 / Accuracy
- Classification report text
- Prediction histogram
- Prediction pie chart
- Simple explainability text
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

# ------------------------------------------------------------
# Helper: Save a matplotlib figure
# ------------------------------------------------------------
def _save_fig_to_png(fig, target_path):
    fig.savefig(target_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return target_path


def _dict_to_kv_lines(d: dict, indent: int = 0):
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(" " * indent + f"{k}:")
            lines.extend(_dict_to_kv_lines(v, indent + 2))
        else:
            lines.append(" " * indent + f"{k}: {v}")
    return lines


# =====================================================================
# TRAIN REPORT
# =====================================================================
def generate_train_report(history, metrics, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    figs = []

    # ------------------------------------------------------
    # LOSS PLOT
    # ------------------------------------------------------
    if "loss" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["loss"], label="Train Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        figs.append(("loss.png", fig))

    # ------------------------------------------------------
    # ACCURACY PLOT
    # ------------------------------------------------------
    if "accuracy" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["accuracy"], label="Train Acc")
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Val Acc")
        ax.set_title("Accuracy Curve")
        ax.legend()
        figs.append(("accuracy.png", fig))

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------
    y_true = metrics.get("y_true")
    y_pred = metrics.get("y_pred")
    cm_path = None
    cls_report_path = None

    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        cm_path = os.path.join(plot_dir, "confusion_matrix.png")
        _save_fig_to_png(fig, cm_path)

        # Classification Report
        cls_report = classification_report(y_true, y_pred)
        cls_report_path = os.path.join(output_dir, "classification_report.txt")

        with open(cls_report_path, "w") as f:
            f.write(cls_report)

    # Save plots
    image_paths = []
    for fname, fig in figs:
        path = os.path.join(plot_dir, fname)
        _save_fig_to_png(fig, path)
        image_paths.append(path)

    # ------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------
    # BUILD PDF
    # ------------------------------------------------------
    pdf_path = os.path.join(output_dir, f"{model_name}_train_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)

    # Title
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ExplainDL — Training Report", ln=True, align="C")
    pdf.ln(4)

    # Metrics Summary
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Metrics Summary", ln=True)
    pdf.set_font("Arial", "", 10)

    for line in _dict_to_kv_lines(metrics):
        pdf.multi_cell(0, 6, line)

    # Add plots
    for img in image_paths:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, os.path.basename(img), ln=True)
        pdf.image(img, x=15, w=180)

    # Confusion matrix
    if cm_path:
        pdf.add_page()
        pdf.cell(0, 8, "Confusion Matrix", ln=True)
        pdf.image(cm_path, x=15, w=180)

    # Classification Report
    if cls_report_path:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, "Classification Report", ln=True)
        pdf.set_font("Arial", "", 10)
        with open(cls_report_path) as f:
            for line in f:
                pdf.multi_cell(0, 5, line)

    pdf.output(pdf_path)
    return pdf_path


# =====================================================================
# PREDICTION REPORT
# =====================================================================
def generate_predict_report(predictions, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    preds = np.array(predictions)
    unique, counts = np.unique(preds, return_counts=True)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(unique, counts)
    ax.set_title("Prediction Histogram")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    hist_path = os.path.join(plot_dir, "prediction_hist.png")
    _save_fig_to_png(fig, hist_path)

    # Pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [class_names[i] if class_names else str(i) for i in unique]
    ax.pie(counts, labels=labels, autopct="%1.1f%%")
    ax.set_title("Prediction Distribution")
    pie_path = os.path.join(plot_dir, "prediction_pie.png")
    _save_fig_to_png(fig, pie_path)

    # ------------------------------------------------------
    # NON-TECHNICAL SUMMARY (IMPROVED)
    # ------------------------------------------------------
    explanation_path = os.path.join(output_dir, "prediction_explanation.txt")

    most_class = unique[np.argmax(counts)]
    most_class_name = (
        class_names[int(most_class)] if class_names and int(most_class) < len(class_names)
        else str(most_class)
    )

    dominance_ratio = counts.max() / counts.sum()

    with open(explanation_path, "w") as f:
        f.write("ExplainDL — Simple Prediction Explanation\n")
        f.write("=========================================\n\n")

        f.write(f"Total Samples: {len(preds)}\n")
        f.write(f"Most Frequent Predicted Class: {most_class_name} ({counts.max()} samples)\n")
        f.write(f"Class Distribution: {dict(zip(unique.tolist(), counts.tolist()))}\n\n")

        # Behaviour interpretation
        f.write("Interpretation:\n")
        if dominance_ratio > 0.85:
            f.write(
                "- The model predicts one class for almost all samples.\n"
                "- This usually indicates **class imbalance** or insufficient feature variation.\n"
                "- You may consider adding more training samples from other classes.\n\n"
            )
        elif dominance_ratio > 0.55:
            f.write(
                "- One class appears more likely than others, but not overwhelmingly.\n"
                "- This suggests the model has learned stronger patterns for this class.\n"
                "- However, it still identifies other classes reasonably.\n\n"
            )
        else:
            f.write(
                "- Predictions are fairly well distributed across classes.\n"
                "- The model seems balanced and confident for multiple patterns.\n\n"
            )

        # Class meaning explanation
        f.write("What this means for your data:\n")
        f.write(
            "- The system groups similar samples based on patterns learned during training.\n"
            "- Higher counts for one class mean your input data resembles that category.\n"
            "- If class names represent categories (e.g., 'spam', 'cat', 'positive'),\n"
            "  they help users understand the meaning of predictions.\n\n"
        )

        # Next step suggestions
        f.write("Suggestions for Non-Technical Users:\n")
        f.write(
            "- Use the prediction distribution chart to visually understand class frequency.\n"
            "- If one class dominates unexpectedly, consider improving dataset balance.\n"
            "- You may upload more diverse examples for better training.\n"
            "- For business decisions, always examine multiple samples before concluding.\n"
        )


    # PDF Build
    pdf_path = os.path.join(output_dir, "predict_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)

    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "ExplainDL — Prediction Report", ln=True, align="C")
    pdf.ln(6)

    # Histogram
    pdf.add_page()
    pdf.cell(0, 8, "Prediction Histogram", ln=True)
    pdf.image(hist_path, x=15, w=180)

    # Pie chart
    pdf.add_page()
    pdf.cell(0, 8, "Prediction Pie Chart", ln=True)
    pdf.image(pie_path, x=15, w=180)

    # Explanation
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Explanation", ln=True)
    pdf.set_font("Arial", "", 10)
    with open(explanation_path) as f:
        for line in f:
            pdf.multi_cell(0, 6, line)

    pdf.output(pdf_path)
    return pdf_path
