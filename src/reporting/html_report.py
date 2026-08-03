"""Self-contained HTML report generator for AutoDL (F1).

Uses Jinja2 (embedded template, no filesystem dependency) and Plotly (CDN)
for interactive charts. Output is a single .html file that opens standalone
in any browser.

Report sections:
  - Run info (model ID, timestamp, seed, dataset type, sample count)
  - Data quality summary
  - Target detection details
  - Model comparison table (all candidates, CV scores, winner highlighted)
  - Final metrics
  - Confusion matrix heatmap (Plotly)
  - Binary threshold (if applicable)
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Embedded template (no filesystem dependency)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AutoDL — Training Report</title>
<style>
  :root {
    --primary: #3b82f6;
    --primary-light: #dbeafe;
    --bg: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --border: #e5e7eb;
    --ok-bg: #dcfce7; --ok-text: #166534;
    --warn-bg: #fef9c3; --warn-text: #854d0e;
    --err-bg: #fee2e2; --err-text: #991b1b;
    --winner-bg: #dbeafe;
    --table-stripe: #f9fafb;
    --table-header: #f0f4ff;
    --code-bg: #f3f4f6;
    --radius: 8px;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      'Helvetica Neue', Arial, sans-serif;
    max-width: 1024px; margin: 0 auto; padding: 2rem 1.5rem;
    color: var(--text); background: var(--bg);
    line-height: 1.6;
  }
  h1 { border-bottom: 3px solid var(--primary); padding-bottom: 0.5rem; }
  h2 { color: var(--primary); margin-top: 2.5rem; }
  h3 { color: #374151; margin-top: 1.5rem; }
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.75rem; margin: 1rem 0;
  }
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem; margin: 1rem 0;
  }
  .meta-item {
    background: var(--table-stripe); border-radius: var(--radius);
    padding: 0.6rem 1rem; border: 1px solid var(--border);
  }
  .meta-item .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; }
  .meta-item .value { font-weight: 600; font-size: 1.05rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; border-radius: var(--radius); overflow: hidden; }
  th, td { border: 1px solid var(--border); padding: 10px 12px; text-align: left; font-size: 0.92rem; }
  th { background: var(--table-header); color: var(--text); font-weight: 600; }
  tbody tr:nth-child(even) { background: var(--table-stripe); }
  .winner-row { background: var(--winner-bg) !important; font-weight: bold; border-left: 4px solid #f59e0b; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 500; }
  .badge-ok { background: var(--ok-bg); color: var(--ok-text); }
  .badge-warn { background: var(--warn-bg); color: var(--warn-text); }
  .badge-err { background: var(--err-bg); color: var(--err-text); }
  .badge-info { background: var(--primary-light); color: #1e40af; }
  .footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); font-size: 0.85rem; color: var(--muted); text-align: center; }
  .section { margin: 1.5rem 0; }
  pre { background: var(--code-bg); padding: 1rem; border-radius: var(--radius); overflow-x: auto; font-size: 0.9rem; }
  .chart-container { margin: 1.5rem 0; display: flex; justify-content: center; }
  .reason-box {
    background: var(--primary-light); border-left: 4px solid var(--primary);
    padding: 0.75rem 1rem; border-radius: 4px; margin: 1rem 0;
    font-size: 0.93rem;
  }
  .threshold-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac; border-radius: var(--radius);
    padding: 1rem 1.25rem; margin: 1rem 0;
    display: flex; align-items: center; gap: 1rem;
  }
  .threshold-value { font-size: 2rem; font-weight: 700; color: #166534; }
  .threshold-detail { font-size: 0.9rem; color: #15803d; }
</style>
</head>
<body>

<h1>AutoDL &mdash; Training Report</h1>

<!-- ====== Summary / Narrative ====== -->
{% if narrative %}
<h2>Summary</h2>
<div class="section">
  <div class="reason-box">
    <p style="margin:0; font-size:1rem; line-height:1.6;">{{ narrative }}</p>
  </div>
</div>
{% endif %}

<!-- ====== Feature Importance ====== -->
{% if feature_importance %}
<h2>Feature Importance</h2>
<div class="section">
  <p style="font-size:0.9rem; color:var(--muted);">
    Shows how much model accuracy drops when each feature is randomly shuffled.
    Longer bars = more important features.
  </p>
  <div class="chart-container">
    {{ feat_importance_chart }}
  </div>
</div>
{% elif feature_importance_unavailable %}
<h2>Feature Importance</h2>
<div class="section">
  <span class="badge badge-warn">Not Available</span>
  <p style="margin-top:0.5rem; color:var(--muted);">
    Feature importance not available for this model type.
  </p>
</div>
{% endif %}

<!-- ====== Run Info ====== -->
<h2>Run Information</h2>
<div class="meta-grid">
  <div class="meta-item"><span class="label">Run ID</span><br><span class="value">{{ meta.model_id }}</span></div>
  <div class="meta-item"><span class="label">Timestamp</span><br><span class="value">{{ meta.timestamp }}</span></div>
  <div class="meta-item"><span class="label">Random Seed</span><br><span class="value">{{ meta.seed }}</span></div>
  <div class="meta-item"><span class="label">Dataset Type</span><br><span class="value">{{ meta.dataset_type }}</span></div>
  <div class="meta-item"><span class="label">Samples</span><br><span class="value">{{ meta.n_samples }}</span></div>
  <div class="meta-item"><span class="label">Classes</span><br><span class="value">{{ meta.n_classes }}</span></div>
</div>

<!-- ====== Dataset Preview ====== -->
{% if dataset_preview and dataset_preview.rows %}
<h2>Dataset Preview (First 5 Rows)</h2>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead>
      <tr>
        {% for col in dataset_preview.columns %}
        <th>{{ col }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in dataset_preview.rows %}
      <tr>
        {% for col in dataset_preview.columns %}
        <td>{{ row[col] }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

<!-- ====== Data Quality ====== -->
<h2>Data Quality</h2>
<div class="section">
{% if quality and quality.summary %}
  {% if quality.passed %}
    <span class="badge badge-ok">All Clear</span>
    <p>{{ quality.summary }}</p>
  {% else %}
    <span class="badge badge-warn">{{ quality.warnings|length }} Issue(s)</span>
    <p>{{ quality.summary }}</p>
    <table>
      <thead><tr><th>Column / Class</th><th>Issue</th><th>Detail</th></tr></thead>
      <tbody>
      {% for w in quality.warnings %}
      <tr>
        <td>{{ w.column or w.class }}</td>
        <td><span class="badge badge-warn">{{ w.issue }}</span></td>
        <td>{{ w.detail }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  {% endif %}
{% else %}
  <p>No quality data available.</p>
{% endif %}
</div>

<!-- ====== Target Detection ====== -->
{% if target %}
<h2>Target Detection</h2>
<div class="section">
  <table>
    <tr><th>Selected Target</th><td><strong>{{ target.column }}</strong></td></tr>
    <tr><th>Status</th><td>
      <span class="badge badge-ok">
        {{ target.status | replace('_',' ') | title }}
      </span>
    </td></tr>
    <tr><th>TLS Score</th><td>{{ "%.4f"|format(target.score) if target.score is number else target.score }}</td></tr>
  </table>
  {% if target.all_scores %}
  <h3>All Column Scores (ranked — top first)</h3>
  <table>
    <thead><tr><th>Rank</th><th>Column</th><th>Score</th><th>Name Signal</th><th>Cardinality Signal</th><th># Unique</th></tr></thead>
    <tbody>
    {% for s in target.all_scores %}
    <tr class="{{ 'winner-row' if s.col == target.column else '' }}">
      <td>{{ loop.index }}</td>
      <td><strong>{{ s.col }}</strong></td>
      <td>{{ "%.4f"|format(s.score) }}</td>
      <td>{{ "%.2f"|format(s.name_score) }}</td>
      <td>{{ "%.2f"|format(s.card_score) }}</td>
      <td>{{ s.n_unique }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endif %}

<!-- ====== Model Comparison ====== -->
<h2>Model Comparison</h2>
<div class="section">
  <table id="model-compare-table">
    <thead>
      <tr>
        <th>Rank</th><th>Model</th><th>CV Score</th><th>Stage</th>
        <th>Description</th><th>Parameters</th><th>Pros</th><th>Cons</th>
      </tr>
    </thead>
    <tbody>
    {% for m in model_compare %}
    <tr class="{{ 'winner-row' if m.name == winner else '' }}">
      <td>{{ loop.index }}</td>
      <td>{{ m.name }}{{ ' [Selected]' if m.name == winner else '' }}</td>
      <td>{{ "%.4f"|format(m.score) if m.score is float else m.score }}</td>
      <td><span class="badge badge-info">Stage {{ m.stage }}</span></td>
      <td>{{ m.description | truncate(80) }}</td>
      <td>{{ m.params | truncate(60) }}</td>
      <td>{{ m.pros | truncate(40) }}</td>
      <td>{{ m.cons | truncate(40) }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  <div class="reason-box">
    <strong>Winner:</strong> <em>{{ winner }}</em> &mdash; {{ winner_reason }}
  </div>
</div>

<!-- ====== Final Metrics ====== -->
<h2>Final Metrics (on held-out validation set)</h2>
<div class="section">
  {% if metrics_summary %}
  <div class="metric-grid">
    {% for label, val in metrics_summary.items() %}
    <div class="meta-item" style="text-align: center; padding: 1rem;">
      <div class="label">{{ label }}</div>
      <div class="value" style="font-size: 1.75rem; color: var(--primary); margin-top: 0.25rem;">
        {% if val is float %}{{ "%.4f"|format(val) }}{% else %}{{ val }}{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
    {% for k, v in metrics.items() %}
    <tr><td>{{ k | replace('_',' ') | title }}</td><td>
      {% if v is float %}{{ "%.4f"|format(v) }}{% else %}{{ v }}{% endif %}
    </td></tr>
    {% endfor %}
    </tbody>
  </table>
  {% if metrics.classification_report %}
  <details>
    <summary style="cursor:pointer; color:var(--primary);">Full Classification Report (per-class)</summary>
    <pre>{{ metrics.classification_report }}</pre>
  </details>
  {% endif %}
</div>

<!-- ====== Confusion Matrix ====== -->
{% if conf_mat %}
<h2>Confusion Matrix</h2>
<div class="section chart-container">
  {{ conf_mat }}
</div>
{% endif %}

<!-- ====== Training Curves ====== -->
{% if loss_plot_base64 or acc_plot_base64 %}
<h2>Training Curves</h2>
<div class="section" style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap;">
  {% if loss_plot_base64 %}
  <div style="text-align: center;">
    <h4 style="margin-bottom: 0.5rem; color: #374151;">Loss Curve</h4>
    <img src="data:image/png;base64,{{ loss_plot_base64 }}" alt="Loss Curve" style="max-width: 450px; width: 100%; height: auto; border-radius: var(--radius); border: 1px solid var(--border);" />
  </div>
  {% endif %}
  {% if acc_plot_base64 %}
  <div style="text-align: center;">
    <h4 style="margin-bottom: 0.5rem; color: #374151;">Accuracy Curve</h4>
    <img src="data:image/png;base64,{{ acc_plot_base64 }}" alt="Accuracy Curve" style="max-width: 450px; width: 100%; height: auto; border-radius: var(--radius); border: 1px solid var(--border);" />
  </div>
  {% endif %}
</div>
{% endif %}

{% if binary_threshold %}
<!-- ====== Binary Threshold ====== -->
<h2>Binary Decision Threshold</h2>
<div class="section">
  <div class="threshold-box">
    <div class="threshold-value">{{ "%.4f"|format(binary_threshold.threshold) }}</div>
    <div class="threshold-detail">
      Optimal threshold via <strong>{{ binary_threshold.strategy | title }}</strong><br>
      Youden's J = {{ "%.4f"|format(binary_threshold.j) }}, AUC = {{ "%.4f"|format(binary_threshold.auc) }}
    </div>
  </div>
  <p style="font-size:0.88rem; color:var(--muted);">
    Predicted probabilities ≥ {{ "%.4f"|format(binary_threshold.threshold) }} are classified as positive.
    This threshold replaces the default 0.5 cutoff.
  </p>
</div>
{% endif %}

<div class="footer">
  <p>AutoDL &mdash; Report generated {{ meta.timestamp }}. Model: <strong>{{ winner }}</strong>. &copy; AutoDL Contributors.</p>
</div>

</body>
</html>"""


def _build_confusion_matrix_fig(
    confusion_matrix: List[List[float]], class_names: Optional[List[str]] = None,
) -> go.Figure:
    """Build a Plotly annotated heatmap Figure for the confusion matrix.
    Returns a go.Figure object suitable for st.plotly_chart().
    """
    n = len(confusion_matrix)
    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]

    # Truncate long labels for display
    display_labels = [(lbl[:18] + "…") if len(lbl) > 20 else lbl for lbl in labels]

    # Calculate annotations (values inside cells)
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=display_labels[j], y=display_labels[i],
                text=str(val),
                showarrow=False,
                font=dict(
                    color="white" if val > max(max(r) for r in confusion_matrix) / 2 else "black",
                    size=13,
                ),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=display_labels,
        y=display_labels,
        colorscale="Blues",
        showscale=True,
        hoverongaps=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=550,
        height=520,
        margin=dict(l=60, r=30, t=50, b=80),
        xaxis=dict(tickangle=-30 if max(len(l) for l in display_labels) > 10 else 0),
        annotations=annotations,
    )
    return fig


def _build_confusion_matrix_plot(
    confusion_matrix: List[List[float]], class_names: Optional[List[str]] = None,
) -> str:
    """Build a Plotly annotated heatmap for the confusion matrix."""
    n = len(confusion_matrix)
    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]

    # Truncate long labels for display
    display_labels = [(lbl[:18] + "…") if len(lbl) > 20 else lbl for lbl in labels]

    # Calculate annotations (values inside cells)
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=display_labels[j], y=display_labels[i],
                text=str(val),
                showarrow=False,
                font=dict(
                    color="white" if val > max(max(r) for r in confusion_matrix) / 2 else "black",
                    size=13,
                ),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=display_labels,
        y=display_labels,
        colorscale="Blues",
        showscale=True,
        hoverongaps=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=550,
        height=520,
        margin=dict(l=60, r=30, t=50, b=80),
        xaxis=dict(tickangle=-30 if max(len(l) for l in display_labels) > 10 else 0),
        annotations=annotations,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_feature_importance_plot(
    feature_importance: List[Dict[str, float]], top_n: int = 10,
) -> str:
    """Build a horizontal bar chart of top-N permutation feature importances."""
    items = feature_importance[:top_n]
    # Reverse so highest is at the top
    items_rev = list(reversed(items))
    features = [fi["feature"] for fi in items_rev]
    importances = [fi["importance"] for fi in items_rev]
    stds = [fi["std"] for fi in items_rev]

    # Truncate long names
    display_labels = [(f[:30] + "…") if len(f) > 32 else f for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=display_labels,
        x=importances,
        orientation="h",
        error_x=dict(
            type="data",
            array=stds,
            visible=True,
            color="#94a3b8",
        ),
        marker=dict(
            color="#3b82f6",
            line=dict(color="#2563eb", width=0.5),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importance: %{x:.4f}<br>"
            "Std: %{error_x.array:.4f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text="Feature Importance (Permutation)",
            font=dict(size=16),
        ),
        xaxis_title="Mean Importance (decrease in accuracy)",
        height=max(300, 50 * len(items_rev)),
        width=750,
        margin=dict(l=10, r=30, t=50, b=60),
        showlegend=False,
        plot_bgcolor="#f9fafb",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_html(template_vars: Dict[str, Any]) -> str:
    """Render the Jinja2 template with the given variables."""
    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(**template_vars)


def generate_html_report(
    meta: Dict,
    quality: Optional[Dict] = None,
    target_info: Optional[Dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Generate a self-contained HTML report; write to output_dir.

    Args:
        meta: full metadata dict from meta.json (or pipeline return). Must contain
              model_id, dataset_type, seed, n_samples, model_comparison, metrics,
              model_name, class_names; optionally binary_threshold.
        quality: quality summary from quality/summarize.py
            {"warnings": [...], "passed": bool, "summary": str}.
        target_info: dict with keys column, status, score, and optionally all_scores.
        output_dir: directory to write report.html (defaults to model_registry/<model_id>).

    Returns:
        Path to the generated report.html file.
    """
    run_dir = output_dir or os.path.join("model_registry", meta.get("model_id", "unknown"))
    os.makedirs(run_dir, exist_ok=True)

    # ---- Confusion Matrix ----
    cm_data = meta.get("metrics", {}).get("confusion_matrix", None)
    cm_html = None
    if cm_data:
        try:
            cm_html = _build_confusion_matrix_plot(
                cm_data, meta.get("class_names", None),
            )
        except Exception as e:
            logger.warning("Failed to build confusion matrix plot: %s", e)

    # ---- Model Comparison ----
    comparison = meta.get("model_comparison", {})
    models = comparison.get("models", [])
    winner = comparison.get("selected", meta.get("model_name", "Unknown"))
    winner_reason = comparison.get("reason", "")

    # ---- Metrics ----
    raw_metrics = meta.get("metrics", {})
    display_metrics = {
        k: v for k, v in raw_metrics.items()
        if k not in ("y_true", "y_pred", "confusion_matrix", "classification_report", "y_pred_proba")
    }
    # Include classification_report as text
    cls_report_dict = raw_metrics.get("classification_report", None)
    if cls_report_dict and isinstance(cls_report_dict, dict):
        # Format as pretty text
        import io
        buf = io.StringIO()
        for cls_key, cls_metrics in cls_report_dict.items():
            if isinstance(cls_metrics, dict):
                buf.write(f"{cls_key}:\n")
                for mk, mv in cls_metrics.items():
                    buf.write(f"  {mk}: {mv:.4f}\n" if isinstance(mv, float) else f"  {mk}: {mv}\n")
            else:
                buf.write(f"{cls_key}: {cls_metrics}\n")
        display_metrics["classification_report"] = buf.getvalue()

    # ---- Binary Threshold ----
    binary_threshold = None
    raw_thr = meta.get("binary_threshold")
    if raw_thr is not None:
        from src.training.threshold import optimize_threshold
        binary_threshold = {
            "threshold": float(raw_thr),
            "strategy": meta.get("binary_threshold_strategy", "youden"),
            "j": meta.get("binary_threshold_j", 0.0),
            "auc": meta.get("binary_threshold_auc", 0.0),
        }

    # ---- Feature Importance Chart ----
    feat_importance_data = meta.get("feature_importance")
    feat_importance_chart = None
    feature_importance_unavailable = False
    narrative_text = meta.get("narrative")

    if feat_importance_data:
        try:
            feat_importance_chart = _build_feature_importance_plot(
                feat_importance_data, top_n=10,
            )
        except Exception as e:
            logger.warning("Failed to build feature importance chart: %s", e)

    # Detect whether we should show "not available" — only for tabular where
    # we attempted it but got None (don't show for image/text where we never try).
    if feat_importance_data is None and meta.get("dataset_type") == "tabular":
        feature_importance_unavailable = True

    # ---- Loss & Accuracy Plots (Base64) ----
    import base64
    loss_plot_base64 = None
    acc_plot_base64 = None
    plots_dir = os.path.join(run_dir, "plots")
    loss_path = os.path.join(plots_dir, "loss.png")
    acc_path = os.path.join(plots_dir, "accuracy.png")
    if os.path.exists(loss_path):
        try:
            with open(loss_path, "rb") as f:
                loss_plot_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning("Failed to load loss plot: %s", e)
    if os.path.exists(acc_path):
        try:
            with open(acc_path, "rb") as f:
                acc_plot_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning("Failed to load accuracy plot: %s", e)

    # ---- Key Metrics Summary Cards ----
    metrics_summary = {}
    for mk in ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]:
        if mk in raw_metrics and isinstance(raw_metrics[mk], (int, float)):
            metrics_summary[mk.replace("_", " ").title()] = raw_metrics[mk]

    # ---- Template Variables ----
    template_vars = {
        "dataset_preview": meta.get("dataset_preview"),
        "loss_plot_base64": loss_plot_base64,
        "acc_plot_base64": acc_plot_base64,
        "metrics_summary": metrics_summary,
        "meta": {
            "model_id": meta.get("model_id", "unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": meta.get("seed", "N/A"),
            "dataset_type": meta.get("dataset_type", "unknown"),
            "n_samples": meta.get("n_samples", "N/A"),
            "n_classes": meta.get("n_classes", meta.get("class_names", []) and len(meta["class_names"]) or "N/A"),
        },
        "quality": quality,
        "target": target_info,
        "narrative": narrative_text,
        "feature_importance": feat_importance_data,
        "feat_importance_chart": feat_importance_chart,
        "feature_importance_unavailable": feature_importance_unavailable,
        "model_compare": models,
        "winner": winner,
        "winner_reason": winner_reason,
        "metrics": display_metrics,
        "conf_mat": cm_html,
        "binary_threshold": binary_threshold,
    }

    html = _build_html(template_vars)

    # ---- Wrap in full HTML with Plotly CDN ----
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
{html}
</body>
</html>"""

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("HTML report written to %s", report_path)
    return report_path