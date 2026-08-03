"""
AutoDL – Extended Report Generator
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

from src.utils.logger import get_logger

logger = get_logger(__name__)

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
def generate_train_report(history, metrics, model_name, output_dir, model_comparison=None, 
                          selection_explanation_path=None, training_explanation_path=None):
    """
    Enhanced training report generator that includes:
    - Model selection details and comparison
    - Selection explanation
    - Training explanation
    - All metrics and visualizations
    """
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
    # BUILD ENHANCED PDF
    # ------------------------------------------------------
    pdf_path = os.path.join(output_dir, f"{model_name}_train_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 15, "ExplainDL - Comprehensive Training Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Model: {model_name}", ln=True, align="C")
    pdf.ln(10)

    # Model Selection Section
    if model_comparison:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "1. Model Selection", ln=True)
        pdf.set_font("Arial", "", 10)
        
        selected = model_comparison.get("selected", "Unknown")
        reason = model_comparison.get("reason", "No reason provided")
        
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"Selected Model: {selected}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"Reason: {reason}")
        pdf.ln(5)
        
        # Model Comparison Table
        if model_comparison.get("models"):
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Model Comparison:", ln=True)
            pdf.set_font("Arial", "", 9)
            
            for model_info in model_comparison["models"]:
                pdf.ln(3)
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 6, f"- {model_info['name']} (Score: {model_info['score']:.4f})", ln=True)
                pdf.set_font("Arial", "", 9)
                pdf.multi_cell(0, 5, f"  {model_info['description']}")
                pdf.multi_cell(0, 5, f"  Parameters: {model_info['params']}")
                pdf.multi_cell(0, 5, f"  Pros: {model_info['pros']}")
                pdf.multi_cell(0, 5, f"  Cons: {model_info['cons']}")
                pdf.ln(2)
        
        pdf.ln(5)

    # Selection Explanation
    if selection_explanation_path and os.path.exists(selection_explanation_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Model Selection Explanation", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.ln(3)
        try:
            with open(selection_explanation_path, "r", encoding="utf-8") as f:
                for line in f:
                    pdf.multi_cell(0, 6, line.strip())
        except Exception:
            pass
        pdf.ln(5)

    # Training Explanation
    if training_explanation_path and os.path.exists(training_explanation_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Training Process Explanation", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.ln(3)
        try:
            with open(training_explanation_path, "r", encoding="utf-8") as f:
                for line in f:
                    pdf.multi_cell(0, 6, line.strip())
        except Exception:
            pass
        pdf.ln(5)

    # Metrics Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "4. Performance Metrics", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", "", 10)

    # Filter out large arrays for PDF
    metrics_for_pdf = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred", "confusion_matrix", "classification_report")}
    
    for line in _dict_to_kv_lines(metrics_for_pdf):
        pdf.multi_cell(0, 6, line)

    # Add plots
    for img in image_paths:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, os.path.basename(img).replace(".png", "").replace("_", " ").title(), ln=True)
        pdf.image(img, x=15, w=180)

    # Confusion matrix
    if cm_path:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Confusion Matrix", ln=True)
        pdf.image(cm_path, x=15, w=180)

    # Classification Report
    if cls_report_path:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, "Classification Report", ln=True)
        pdf.set_font("Arial", "", 9)
        try:
            with open(cls_report_path) as f:
                for line in f:
                    pdf.multi_cell(0, 5, line.strip())
        except Exception:
            pass

    pdf.output(pdf_path)
    return pdf_path


# =====================================================================
# PREDICTION REPORT
# =====================================================================
def generate_predict_report(predictions, class_names, output_dir, dataset_type,
                            model_name="Unknown", meta=None, prediction_probas=None):
    """Generate a detailed prediction report PDF + explanation text.

    `output_dir` is the model_dir (absolute path) — no relative-path traversal.
    `dataset_type` is passed directly from metadata by the caller.
    `meta` is the full model metadata dict (optional, for richer context).
    `prediction_probas` is the probability matrix (N x num_classes) for confidence analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    preds = np.array(predictions)
    n_samples = len(preds)
    unique, counts = np.unique(preds, return_counts=True)
    n_classes = len(unique)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ----------------------------------------------------------
    # VISUALIZATIONS
    # ----------------------------------------------------------

    # 1. Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(unique, counts, edgecolor='black', alpha=0.7)
    ax.set_title("Prediction Histogram")
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Count")
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=9)
    hist_path = os.path.join(plot_dir, "prediction_hist.png")
    _save_fig_to_png(fig, hist_path)

    # 2. Pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [class_names[i] if class_names else str(i) for i in unique]
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title("Prediction Distribution")
    pie_path = os.path.join(plot_dir, "prediction_pie.png")
    _save_fig_to_png(fig, pie_path)

    # 3. Confidence distribution (if probabilities available)
    conf_path = None
    class_conf_paths = {}
    if prediction_probas is not None and prediction_probas.ndim == 2:
        # Overall confidence (max probability per sample)
        max_probas = np.max(prediction_probas, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(max_probas, bins=20, edgecolor='black', alpha=0.7, range=(0, 1))
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Max Probability (Confidence)")
        ax.set_ylabel("Number of Samples")
        ax.axvline(np.mean(max_probas), color='red', linestyle='--',
                   label=f'Mean: {np.mean(max_probas):.3f}')
        ax.legend()
        conf_path = os.path.join(plot_dir, "confidence_dist.png")
        _save_fig_to_png(fig, conf_path)

        # Per-class confidence distributions
        for cls_idx in unique:
            cls_mask = preds == cls_idx
            if np.any(cls_mask):
                cls_probas = prediction_probas[cls_mask, cls_idx]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(cls_probas, bins=min(15, len(cls_probas)), edgecolor='black', alpha=0.7, range=(0, 1))
                cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                ax.set_title(f"Confidence for Class: {cls_name}")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Count")
                if len(cls_probas) > 0:
                    ax.axvline(np.mean(cls_probas), color='red', linestyle='--',
                               label=f'Mean: {np.mean(cls_probas):.3f}')
                    ax.legend()
                class_conf_paths[cls_idx] = os.path.join(plot_dir, f"confidence_class_{cls_idx}.png")
                _save_fig_to_png(fig, class_conf_paths[cls_idx])

    # 4. Class-wise count bar chart (horizontal, with names)
    fig, ax = plt.subplots(figsize=(7, max(3, n_classes * 0.4)))
    y_pos = np.arange(n_classes)
    bar_labels = [class_names[int(i)] if class_names and int(i) < len(class_names) else str(i) for i in unique]
    bars = ax.barh(y_pos, counts, edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Per-Class Prediction Counts")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontsize=9)
    class_counts_path = os.path.join(plot_dir, "class_counts.png")
    _save_fig_to_png(fig, class_counts_path)

    # ----------------------------------------------------------
    # TEXT EXPLANATION (DETAILED)
    # ----------------------------------------------------------
    explanation_path = os.path.join(output_dir, "prediction_explanation.txt")

    most_class = unique[np.argmax(counts)]
    most_count = counts.max()
    least_class = unique[np.argmin(counts)]
    least_count = counts.min()
    most_class_name = class_names[int(most_class)] if class_names and int(most_class) < len(class_names) else str(most_class)
    least_class_name = class_names[int(least_class)] if class_names and int(least_class) < len(class_names) else str(least_class)

    dominance_ratio = most_count / n_samples
    entropy = -np.sum((counts / n_samples) * np.log2(counts / n_samples + 1e-10))
    max_entropy = np.log2(n_classes) if n_classes > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    def label_map(idx):
        return class_names[idx] if class_names and idx < len(class_names) else str(idx)

    # Confidence stats
    conf_stats = {}
    if prediction_probas is not None and prediction_probas.ndim == 2:
        max_probas = np.max(prediction_probas, axis=1)
        conf_stats = {
            'mean': float(np.mean(max_probas)),
            'median': float(np.median(max_probas)),
            'std': float(np.std(max_probas)),
            'min': float(np.min(max_probas)),
            'q25': float(np.percentile(max_probas, 25)),
            'q75': float(np.percentile(max_probas, 75)),
            'high_conf_pct': float(np.mean(max_probas > 0.9) * 100),
            'low_conf_pct': float(np.mean(max_probas < 0.6) * 100),
        }

    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AutoDL - Detailed Prediction Report\n")
        f.write("=" * 60 + "\n\n")

        # Model & Dataset Info
        f.write("1. MODEL & DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        if meta:
            f.write(f"   Model ID: {meta.get('model_id', 'Unknown')}\n")
            f.write(f"   Model Name: {meta.get('model_name', model_name)}\n")
            f.write(f"   Dataset Type: {meta.get('dataset_type', dataset_type)}\n")
            f.write(f"   Training Date: {meta.get('timestamp', 'Unknown')}\n")
            f.write(f"   Classes: {', '.join(meta.get('class_names', []))}\n")
            if meta.get('binary_threshold'):
                f.write(f"   Binary Threshold (Youden's J): {meta['binary_threshold']:.4f}\n")
            f.write(f"   Random Seed: {meta.get('seed', 'Unknown')}\n")
        else:
            f.write(f"   Model Name: {model_name}\n")
            f.write(f"   Dataset Type: {dataset_type}\n")
            f.write(f"   Classes: {', '.join([label_map(i) for i in unique])}\n")
        f.write(f"\n")

        # Overall Statistics
        f.write("2. OVERALL PREDICTION STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"   Total Samples: {n_samples}\n")
        f.write(f"   Number of Predicted Classes: {n_classes}\n")
        f.write(f"   Most Frequent Class: {most_class_name} ({most_count} samples, {most_count/n_samples*100:.1f}%)\n")
        f.write(f"   Least Frequent Class: {least_class_name} ({least_count} samples, {least_count/n_samples*100:.1f}%)\n")
        f.write(f"   Dominance Ratio (top class): {dominance_ratio:.2%}\n")
        f.write(f"   Prediction Entropy: {entropy:.3f} / {max_entropy:.3f} (normalized: {normalized_entropy:.2%})\n")
        if conf_stats:
            f.write(f"   Mean Confidence: {conf_stats['mean']:.3f}\n")
            f.write(f"   Median Confidence: {conf_stats['median']:.3f}\n")
            f.write(f"   Confidence Std Dev: {conf_stats['std']:.3f}\n")
            f.write(f"   High Confidence (>0.9): {conf_stats['high_conf_pct']:.1f}%\n")
            f.write(f"   Low Confidence (<0.6): {conf_stats['low_conf_pct']:.1f}%\n")
        f.write(f"\n")

        # Per-Class Breakdown
        f.write("3. PER-CLASS BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        for cls_idx, count in zip(unique, counts):
            cls_name = label_map(int(cls_idx))
            pct = count / n_samples * 100
            f.write(f"   Class {cls_name}: {count} samples ({pct:.1f}%)\n")
            if prediction_probas is not None and cls_idx in class_conf_paths:
                cls_probas = prediction_probas[preds == cls_idx, cls_idx]
                f.write(f"      Mean Confidence: {np.mean(cls_probas):.3f}\n")
                f.write(f"      Std Confidence: {np.std(cls_probas):.3f}\n")
                f.write(f"      Min/Max: {np.min(cls_probas):.3f} / {np.max(cls_probas):.3f}\n")
        f.write(f"\n")

        # Confidence Distribution Details
        if conf_stats:
            f.write("4. CONFIDENCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"   Confidence Range: [{conf_stats['min']:.3f}, 1.000]\n")
            f.write(f"   Interquartile Range: [{conf_stats['q25']:.3f}, {conf_stats['q75']:.3f}]\n")
            f.write(f"   High Confidence (>90%): {conf_stats['high_conf_pct']:.1f}% of samples\n")
            f.write(f"   Low Confidence (<60%): {conf_stats['low_conf_pct']:.1f}% of samples\n")
            if conf_stats['low_conf_pct'] > 20:
                f.write("   WARNING: Many low-confidence predictions - consider reviewing data quality\n")
            elif conf_stats['high_conf_pct'] > 70:
                f.write("   GOOD: Most predictions are high confidence\n")
            f.write(f"\n")

        # Dataset-Type Specific Interpretation
        f.write("5. INTERPRETATION & RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")

        if dataset_type == "tabular":
            f.write("Tabular Data Interpretation:\n")
            if dominance_ratio > 0.85:
                f.write("- Strong class dominance detected. This may indicate:\n")
                f.write("    * Significant class imbalance in training data\n")
                f.write("    * Input features strongly predictive of one class\n")
                f.write("    * Potential data leakage or target encoding issues\n\n")
            elif dominance_ratio > 0.55:
                f.write("- Moderate class dominance. The model finds clear patterns\n")
                f.write("  but multiple classes are represented.\n\n")
            else:
                f.write("- Balanced predictions across classes. The model recognizes\n")
                f.write("  diverse feature patterns in the data.\n\n")

            if conf_stats:
                if conf_stats['low_conf_pct'] > 30:
                    f.write("Low confidence predictions suggest:\n")
                    f.write("    * Input rows may differ significantly from training distribution\n")
                    f.write("    * Feature values may be out-of-distribution\n")
                    f.write("    * Consider collecting more diverse training data\n\n")
                elif conf_stats['high_conf_pct'] > 60:
                    f.write("High confidence predictions indicate:\n")
                    f.write("    * Input features align well with training patterns\n")
                    f.write("    * Model is operating within its domain of competence\n\n")

            f.write("Recommendations:\n")
            f.write("- Verify input columns match training schema\n")
            f.write("- Check for missing values in key features\n")
            f.write("- Consider feature importance analysis to understand drivers\n")
            f.write("- For low-confidence samples, manual review is recommended\n")

        elif dataset_type == "image":
            f.write("Image Data Interpretation:\n")
            if dominance_ratio > 0.7:
                f.write("- Visual patterns appear similar across many images.\n")
                f.write("  This may indicate a homogeneous dataset or strong visual signal.\n\n")
            else:
                f.write("- Diverse visual patterns detected across predictions.\n\n")

            if conf_stats:
                if conf_stats['low_conf_pct'] > 25:
                    f.write("Low confidence may stem from:\n")
                    f.write("    * Blurry, low-light, or noisy images\n")
                    f.write("    * Objects partially occluded or at unusual angles\n")
                    f.write("    * Background clutter interfering with features\n\n")
                else:
                    f.write("High confidence suggests:\n")
                    f.write("    * Clear, well-framed images matching training distribution\n")
                    f.write("    * Distinct visual features for each class\n\n")

            f.write("Recommendations:\n")
            f.write("- Ensure images are clear, well-lit, and properly oriented\n")
            f.write("- Avoid heavy compression artifacts\n")
            f.write("- Use Grad-CAM (if available) to inspect attention regions\n")
            f.write("- For low-confidence images, manual verification advised\n")

        elif dataset_type == "text":
            f.write("Text Data Interpretation:\n")
            if dominance_ratio > 0.75:
                f.write("- Text samples share similar tone, vocabulary, or topic.\n")
                f.write("  This may reflect a homogeneous corpus.\n\n")
            else:
                f.write("- Text contains varied themes, sentiments, or writing styles.\n\n")

            if conf_stats:
                if conf_stats['low_conf_pct'] > 25:
                    f.write("Low confidence may indicate:\n")
                    f.write("    * Short or ambiguous text samples\n")
                    f.write("    * Mixed topics within single samples\n")
                    f.write("    * Vocabulary outside training distribution\n\n")
                else:
                    f.write("High confidence suggests:\n")
                    f.write("    * Clear linguistic patterns matching training classes\n")
                    f.write("    * Sufficient text length for reliable classification\n\n")

            f.write("Recommendations:\n")
            f.write("- Provide longer, clearer text when possible\n")
            f.write("- Ensure text language matches training data\n")
            f.write("- Avoid mixing multiple topics in single samples\n")
            f.write("- For low-confidence samples, manual review recommended\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")

    # ----------------------------------------------------------
    # SAMPLE-LEVEL PREDICTION CSV (for download)
    # ----------------------------------------------------------
    csv_path = os.path.join(output_dir, "predictions_detailed.csv")
    try:
        if prediction_probas is not None and prediction_probas.ndim == 2:
            # Build detailed DataFrame
            rows = []
            for i in range(n_samples):
                row = {
                    'sample_index': i,
                    'predicted_class': int(preds[i]),
                    'predicted_class_name': label_map(int(preds[i])),
                    'confidence': float(np.max(prediction_probas[i])),
                }
                # Add per-class probabilities
                for cls_idx in unique:
                    cls_name = label_map(int(cls_idx)).replace(' ', '_')
                    row[f'prob_{cls_name}'] = float(prediction_probas[i, cls_idx])
                rows.append(row)
            import pandas as pd
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    except Exception as e:
        logger.warning("Could not write detailed predictions CSV: %s", e)

    # ----------------------------------------------------------
    # PDF BUILD
    # ----------------------------------------------------------
    pdf_path = os.path.join(output_dir, "predict_report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 15, "AutoDL - Detailed Prediction Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Model: {model_name}", ln=True, align="C")
    if meta:
        pdf.cell(0, 7, f"Model ID: {meta.get('model_id', 'Unknown')}", ln=True, align="C")
        pdf.cell(0, 7, f"Dataset Type: {meta.get('dataset_type', dataset_type)}", ln=True, align="C")
    pdf.cell(0, 7, f"Total Samples: {n_samples} | Classes: {n_classes}", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 5, "Generated by AutoDL - Local AutoML for Classification", ln=True, align="C")
    pdf.ln(10)

    # Summary Stats Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Summary Statistics", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.ln(2)

    # Table header
    col_w = [80, 110]
    pdf.set_font("Arial", "B", 9)
    pdf.cell(col_w[0], 6, "Metric", 1)
    pdf.cell(col_w[1], 6, "Value", 1, ln=True)
    pdf.set_font("Arial", "", 9)

    rows = [
        ("Total Samples", str(n_samples)),
        ("Predicted Classes", str(n_classes)),
        ("Most Frequent Class", f"{most_class_name} ({most_count}, {most_count/n_samples*100:.1f}%)"),
        ("Least Frequent Class", f"{least_class_name} ({least_count}, {least_count/n_samples*100:.1f}%)"),
        ("Dominance Ratio", f"{dominance_ratio:.2%}"),
        ("Prediction Entropy", f"{entropy:.3f} (normalized: {normalized_entropy:.2%})"),
    ]
    if conf_stats:
        rows.extend([
            ("Mean Confidence", f"{conf_stats['mean']:.3f}"),
            ("Median Confidence", f"{conf_stats['median']:.3f}"),
            ("Confidence Std Dev", f"{conf_stats['std']:.3f}"),
            ("High Confidence (>0.9)", f"{conf_stats['high_conf_pct']:.1f}%"),
            ("Low Confidence (<0.6)", f"{conf_stats['low_conf_pct']:.1f}%"),
        ])
    if meta:
        rows.extend([
            ("Model ID", meta.get('model_id', 'Unknown')),
            ("Training Seed", str(meta.get('seed', 'Unknown'))),
            ("Binary Threshold", f"{meta.get('binary_threshold', 'N/A')}" if meta.get('binary_threshold') else "N/A (multiclass)"),
        ])

    for metric, value in rows:
        pdf.cell(col_w[0], 6, metric, 1)
        pdf.cell(col_w[1], 6, value, 1, ln=True)
    pdf.ln(5)

    # Per-Class Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Per-Class Prediction Counts", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.ln(2)

    col_w2 = [50, 30, 30, 80]
    pdf.set_font("Arial", "B", 9)
    pdf.cell(col_w2[0], 6, "Class", 1)
    pdf.cell(col_w2[1], 6, "Count", 1)
    pdf.cell(col_w2[2], 6, "Percentage", 1)
    if conf_stats:
        pdf.cell(col_w2[3], 6, "Mean Confidence", 1, ln=True)
    else:
        pdf.ln(6)
    pdf.set_font("Arial", "", 9)

    for cls_idx, count in zip(unique, counts):
        cls_name = label_map(int(cls_idx))
        pct = count / n_samples * 100
        pdf.cell(col_w2[0], 6, cls_name[:30], 1)
        pdf.cell(col_w2[1], 6, str(count), 1)
        pdf.cell(col_w2[2], 6, f"{pct:.1f}%", 1)
        if conf_stats and cls_idx in class_conf_paths:
            cls_probas = prediction_probas[preds == cls_idx, cls_idx]
            pdf.cell(col_w2[3], 6, f"{np.mean(cls_probas):.3f}", 1, ln=True)
        else:
            pdf.cell(col_w2[3], 6, "N/A", 1, ln=True)
    pdf.ln(5)

    # Visualizations
    # Histogram
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Prediction Histogram", ln=True)
    pdf.image(hist_path, x=15, w=180)

    # Pie Chart
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Prediction Distribution (Pie)", ln=True)
    pdf.image(pie_path, x=15, w=180)

    # Class Counts Horizontal Bar
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "5. Per-Class Counts", ln=True)
    pdf.image(class_counts_path, x=15, w=180)

    # Confidence Distribution
    if conf_path:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "6. Overall Confidence Distribution", ln=True)
        pdf.image(conf_path, x=15, w=180)

    # Per-class confidence (limit to first 4 to avoid too many pages)
    if class_conf_paths:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "7. Per-Class Confidence Distributions", ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.ln(3)
        for i, (cls_idx, path) in enumerate(list(class_conf_paths.items())[:4]):
            if i > 0:
                pdf.ln(5)
            cls_name = label_map(int(cls_idx))
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, f"Class: {cls_name}", ln=True)
            pdf.image(path, x=15, w=180)

    # Explanation Text
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "8. Interpretation & Recommendations", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.ln(3)
    try:
        with open(explanation_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("=") or line.startswith("-"):
                    continue
                if line:
                    pdf.multi_cell(0, 5, line)
    except Exception:
        pdf.cell(0, 6, "Explanation text not available.", ln=True)

    pdf.output(pdf_path)
    return pdf_path