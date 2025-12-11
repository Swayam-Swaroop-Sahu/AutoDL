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
def generate_predict_report(predictions, class_names, output_dir,dataset_type):
    # Load dataset_type from the trained model’s meta.json
    meta_path = os.path.join(output_dir, "..", "meta.json")
    dataset_type_from_meta = None

    try:
        with open(meta_path, "r") as m:
            meta = json.load(m)
            dataset_type_from_meta = meta.get("dataset_type")
    except:
        dataset_type_from_meta = None

    # Override dataset_type if available
    if dataset_type_from_meta:
        dataset_type = dataset_type_from_meta


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
    # NON-TECHNICAL SUMMARY (DATASET-AWARE EXPLANATION)
    # ------------------------------------------------------
    explanation_path = os.path.join(output_dir, "prediction_explanation.txt")

    most_class = unique[np.argmax(counts)]
    most_class_name = (
        class_names[int(most_class)] if class_names and int(most_class) < len(class_names)
        else str(most_class)
    )

    dominance_ratio = counts.max() / counts.sum()

    def label_map(idx):
        return class_names[idx] if class_names and idx < len(class_names) else str(idx)

    # ======================================================
    # WRITE EXPLANATION BASED ON DATASET TYPE
    # ======================================================
    with open(explanation_path, "w", encoding="utf-8") as f:

        f.write("Simple Prediction Explanation\n")
        f.write("==========================================\n\n")

        f.write(f"Total Samples: {len(preds)}\n")
        f.write(f"Most Frequent Predicted Class: {label_map(int(most_class))} ({counts.max()} samples)\n")
        f.write(f"Class Distribution: { {label_map(int(k)): int(v) for k,v in zip(unique, counts)} }\n\n")

        # --------------------------------------------------
        # TABULAR EXPLANATION
        # --------------------------------------------------
        if dataset_type == "tabular":
            f.write("Interpretation (Tabular Data):\n")
            if dominance_ratio > 0.85:
                f.write("- One class dominates most predictions.\n"
                        "- This often points to strong feature similarity or dataset imbalance.\n\n")
            elif dominance_ratio > 0.55:
                f.write("- A moderate class dominance suggests clear but not overwhelming patterns.\n\n")
            else:
                f.write("- Predictions are well spread; the model recognizes multiple feature patterns.\n\n")

            f.write(
                "How the model made decisions:\n"
                "- It learned relationships between numerical/text features.\n"
                "- Rows with similar feature values tend to receive the same class.\n"
                "- SHAP values (if enabled) highlight which features contributed most.\n\n"
            )

            f.write(
                "Tips for Users:\n"
                "- If predictions seem biased, inspect dataset balance.\n"
                "- Add more diverse examples for underrepresented classes.\n"
                "- Use the confusion matrix to verify class-wise performance.\n"
            )

        # --------------------------------------------------
        # IMAGE EXPLANATION
        # --------------------------------------------------
        if dataset_type == "image":
            f.write("Interpretation (Image Data):\n")
            f.write("- The model identifies objects based on shapes, edges, colors, and textures.\n")
            if dominance_ratio > 0.7:
                f.write("- Many images appear visually similar, leading to one dominant prediction.\n")
            else:
                f.write("- The variation in predictions suggests diverse visual patterns.\n")
            f.write("\n")

            f.write(
                "How the model sees the images:\n"
                "- Convolutional filters detect patterns like corners, textures, and contours.\n"
                "- Grad-CAM highlights which regions influenced each prediction.\n\n"
            )

            f.write(
                "Tips for Users:\n"
                "- Ensure images are clear, well-lit, and non-blurry.\n"
                "- Background noise may cause misclassification.\n"
                "- Provide diverse training examples for higher robustness.\n"
            )

        # --------------------------------------------------
        # TEXT EXPLANATION
        # --------------------------------------------------
        if dataset_type == "text":
            f.write("Interpretation (Text Data):\n")
            if dominance_ratio > 0.75:
                f.write("- The uploaded text samples mostly match the tone/topic of one class.\n")
            else:
                f.write("- The text contains mixed themes or sentiments, leading to varied predictions.\n")
            f.write("\n")

            f.write(
                "How the model understands text:\n"
                "- It analyzes word patterns, key phrases, sentiment cues, and writing style.\n"
                "- LSTM/BiLSTM models detect long-range dependencies.\n"
                "- Text-CNN captures short local patterns.\n\n"
            )

            f.write(
                "Tips for Users:\n"
                "- If predictions seem skewed, check whether samples share similar tone.\n"
                "- Add more topic variety if you want clearer class separation.\n"
                "- Longer text generally produces more reliable predictions.\n"
            )

        
    # ------------------------------------------------------
    # PDF BUILD
    # ------------------------------------------------------
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

    # Explanation section
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Explanation", ln=True)
    pdf.set_font("Arial", "", 10)
    with open(explanation_path, "r", encoding="utf-8") as f:
        for line in f:
            pdf.multi_cell(0, 6, line)

    pdf.output(pdf_path)
    return pdf_path
