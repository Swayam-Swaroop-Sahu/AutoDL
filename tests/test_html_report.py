"""Tests for HTML report generation (F1)."""
import os
import tempfile

import pytest
from bs4 import BeautifulSoup

from src.reporting.html_report import generate_html_report


def _meta_dict():
    return {
        "model_id": "test123",
        "dataset_type": "tabular",
        "model_name": "LogReg",
        "class_names": ["no", "yes"],
        "metrics": {
            "accuracy": 0.88,
            "balanced_accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.86,
            "f1_score": 0.865,
            "confusion_matrix": [[40, 5], [8, 47]],
        },
        "model_comparison": {
            "selected": "LogReg",
            "reason": "won search",
            "models": [
                {"name": "LogReg", "score": 0.85, "description": "LR", "params": ""},
                {"name": "RF", "score": 0.82, "description": "RF", "params": ""},
            ],
        },
        "binary_threshold": 0.423,
        "n_classes": 2,
    }


def test_html_report_created():
    """generate_html_report creates a report.html file."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_01"
        quality = {"passed": True, "warnings": [], "summary": "All clear."}
        target = {"column": "target", "status": "strong_auto", "score": 0.92}
        path = generate_html_report(meta, quality, target, output_dir=tmp)
        assert os.path.exists(path)
        assert path.endswith("report.html")


def test_html_has_all_sections():
    """Report contains all required sections: header, quality, target,
    model comparison, metrics, confusion matrix, threshold, footer."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_02"
        quality = {"warnings": [], "passed": True, "summary": "All clear."}
        target = {"column": "target", "status": "strong_auto", "score": 0.95}
        path = generate_html_report(meta, quality, target, output_dir=tmp)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")

        # Header
        assert "AutoDL" in soup.text
        # Data Quality section
        assert "Data Quality" in soup.text or "All Clear" in soup.text
        # Target Detection section
        assert "Target Detection" in soup.text
        # Model Comparison section
        assert "Model Comparison" in soup.text
        # Final Metrics section
        assert "Final Metrics" in soup.text
        # Confusion Matrix section
        assert "Confusion Matrix" in soup.text
        # Footer
        assert "run_02" in soup.text


def test_html_report_with_warnings_shows_warning_badge():
    """When quality has warnings, a warning badge is rendered."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_03"
        quality = {
            "passed": False,
            "warnings": [
                {"column": "leaky", "issue": "leakage", "detail": "r=0.99 with target."},
            ],
            "summary": "Found 1 issue: leakage.",
        }
        path = generate_html_report(meta, quality, output_dir=tmp)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        assert "leaky" in html
        assert "leakage" in html
        assert "r=0.99" in html


def test_html_report_without_cm_still_works():
    """If confusion_matrix is not present, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_04"
        del meta["metrics"]["confusion_matrix"]
        path = generate_html_report(meta, output_dir=tmp)
        assert os.path.exists(path)


def test_html_no_binary_threshold_for_multiclass():
    """When binary_threshold is None, the section is not rendered."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_05"
        meta["binary_threshold"] = None
        quality = {"warnings": [], "passed": True, "summary": "All clear."}
        path = generate_html_report(meta, quality, output_dir=tmp)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        assert "Binary Threshold" not in html


def test_html_report_includes_dataset_preview():
    """When dataset_preview is in meta, a Dataset Preview table is rendered."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = _meta_dict()
        meta["model_id"] = "run_06"
        meta["dataset_preview"] = {
            "columns": ["feature_a", "feature_b", "target"],
            "rows": [
                {"feature_a": 1.0, "feature_b": 2.0, "target": 0},
                {"feature_a": 3.0, "feature_b": 4.0, "target": 1},
            ],
        }
        path = generate_html_report(meta, output_dir=tmp)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        assert "Dataset Preview (First 5 Rows)" in html
        assert "feature_a" in html
        assert "feature_b" in html


def test_html_report_includes_training_curves_and_metric_cards():
    """When loss.png / accuracy.png exist in plots/, Training Curves are embedded."""
    with tempfile.TemporaryDirectory() as tmp:
        plots_dir = os.path.join(tmp, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        # Create dummy plot files
        with open(os.path.join(plots_dir, "loss.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")
        with open(os.path.join(plots_dir, "accuracy.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")

        meta = _meta_dict()
        meta["model_id"] = "run_07"
        path = generate_html_report(meta, output_dir=tmp)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        assert "Training Curves" in html
        assert "data:image/png;base64," in html
        assert "Accuracy" in html
        assert "Precision" in html
        assert "Recall" in html