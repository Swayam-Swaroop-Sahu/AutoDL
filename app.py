"""
Streamlit UI for AutoDL — Automated Classification & Insights.

User journey:
  1. Upload labelled dataset → preview shown
  2. Target detection runs → ranked TLS table displayed
  3. If ambiguous → st.radio() to confirm target
  4. User clicks "Train Model" → st.status() progress
  5. Post-training → model comparison table, metrics, confusion matrix,
     download report button
  6. Any exception → st.error() with "what-why-what to do" (no traceback)

Exit flows:
  - Empty CSV → clear error message
  - Ambiguous target → ranked table + radio
  - Training → progress bar inside st.status
  - Post-training → comparison table + report download
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import traceback
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from src.core.config import RANDOM_SEED, MODEL_REGISTRY_DIR
from src.core.exceptions import (
    AutoDLInputError,
    AutoDLTrainingError,
    AutoDLTargetAmbiguousError,
)
from src.core.validation import validate_file_exists, validate_non_empty
from src.core.pipeline_train import train_pipeline
from src.core.pipeline_predict import predict_pipeline
from src.registry import register_model
from src.target_detection import rank_target_candidates
from src.data.detect_type import detect_dataset_type
from src.reporting.html_report import _build_confusion_matrix_fig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoDL — Automated Classification",
    page_icon="",
    layout="wide",
)
st.title("AutoDL — Automated Classification & Insights")
st.write("Upload a labelled dataset to train a model, then upload unlabelled data for prediction.")

# ─────────────────────────────────────────────────────────────────────
# Helper: save uploaded file to a temp path
# ─────────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file, prefix: str = "autodl") -> str:
    """Save a Streamlit UploadedFile to a temp path; return path."""
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.read())
    return path


# ─────────────────────────────────────────────────────────────────────
# Helper: Build confusion matrix Plotly figure
# ─────────────────────────────────────────────────────────────────────
def _build_conf_matrix_fig(
    cm: list[list[int]],
    class_names: Optional[list[str]] = None,
) -> go.Figure:
    """Return a Plotly annotated heatmap Figure for the confusion matrix."""
    n = len(cm)
    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    display_labels = [(lbl[:18] + "…") if len(lbl) > 20 else lbl for lbl in labels]

    # Color-coding annotations
    max_val = max(max(row) for row in cm)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=display_labels,
        y=display_labels,
        colorscale="Blues",
        showscale=True,
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))

    # Add annotations
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=display_labels[j],
                y=display_labels[i],
                text=str(val),
                showarrow=False,
                font=dict(
                    color="white" if val > max_val / 2 else "black",
                    size=13,
                ),
            )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=550,
        height=500,
        margin=dict(l=60, r=30, t=50, b=80),
        xaxis=dict(tickangle=-30 if max(len(l) for l in display_labels) > 10 else 0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────
if "model_dir" not in st.session_state:
    st.session_state.model_dir = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "dataset_type" not in st.session_state:
    st.session_state.dataset_type = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "target_scores_df" not in st.session_state:
    st.session_state.target_scores_df = None

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Train Model
# ─────────────────────────────────────────────────────────────────────
st.header("Step 1 — Upload & Train")
train_file = st.file_uploader(
    "Upload labelled dataset (CSV, XLSX, TXT, ZIP for images)",
    type=["csv", "xlsx", "txt", "zip"],
    key="train_file",
)

# ---- Preview (first 5 rows) for tabular files ----
if train_file is not None:
    try:
        train_file.seek(0)
        if train_file.name.lower().endswith((".csv", ".xlsx")):
            if train_file.name.lower().endswith(".csv"):
                df_preview = pd.read_csv(train_file, nrows=5, on_bad_lines="warn")
            else:
                df_preview = pd.read_excel(train_file, nrows=5)
            st.subheader("Preview (first 5 rows)")
            st.dataframe(df_preview, use_container_width=True)
    except Exception:
        st.info("Preview not available for this file type.")

# ---- Detect dataset type early ----
detected_type = None
if train_file is not None:
    try:
        dataset_path_temp = save_uploaded_file(train_file, prefix="detect")
        detected_type = detect_dataset_type(dataset_path_temp)
        st.info(f"**Detected dataset type:** `{detected_type}`")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────
# Target detection for tabular data (runs after file upload before train clicked)
# ─────────────────────────────────────────────────────────────────────
st.subheader("Target Column")
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
target_collected = st.session_state.get('target_col')

if train_file is not None and detected_type == "tabular":
    try:
        # Load the full dataframe for target detection
        dataset_path_temp = save_uploaded_file(train_file, prefix="target")
        from src.data.tabular_loader import load_tabular_data
        df_full = load_tabular_data(dataset_path_temp, require_target=True)
        validate_non_empty(df_full, name="uploaded dataset")

        # Rank every column by target-likelihood (name + cardinality signals only)
        ranked = rank_target_candidates(df_full)
        st.session_state['target_scores_df'] = ranked

        if not ranked:
            st.error(
                "No columns found in the uploaded dataset. "
                "Why: the file may contain only a header. "
                "What to do: add feature columns and re-upload."
            )
        else:
            st.markdown(
                "#### Ranked target candidates "
                "(higher score = more likely the classification target)"
            )
            ranked_df = pd.DataFrame(ranked)
            ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
            st.dataframe(
                ranked_df.style.format({
                    "score": "{:.4f}",
                    "name_score": "{:.2f}",
                    "card_score": "{:.2f}",
                }).background_gradient(subset=["score"], cmap="Blues"),
                use_container_width=True,
                hide_index=True,
            )

            # User MUST pick one. Top candidate is the default option.
            col_options = [c["col"] for c in ranked]
            default_idx = 0
            
            def _fmt(c):
                row = next((r for r in ranked if r["col"] == c), None)
                if row is None:
                    return c
                flag = " (recommended)" if c == col_options[0] else ""
                return f"{c}  (score={row['score']:.3f}, n_unique={row['n_unique']}){flag}"
            
            chosen = st.selectbox(
                "Select the target column to train on:",
                options=col_options,
                index=default_idx,
                format_func=_fmt,
                key="target_select",
            )
            st.session_state['target_col'] = chosen
            target_collected = chosen

            # Warn (but do not block) when the user chose a low-ranked column.
            chosen_row = next((r for r in ranked if r["col"] == chosen), None)
            if chosen_row and chosen_row["score"] < 0.2:
                st.warning(
                    f"Heads up: '{chosen}' has a low target-likelihood score "
                    f"({chosen_row['score']:.3f}). AutoDL will still use it as the target — "
                    "just double-check this is the column you want to predict."
                )
            elif chosen_row:
                st.caption(f"Using target column: `{chosen}` (score {chosen_row['score']:.3f}).")
    except AutoDLInputError as e:
        st.error(f"**Data Error:** {str(e)}")
    except Exception as e:
        st.warning(f"Target detection skipped: {e}")


# ─────────────────────────────────────────────────────────────────────
# Model Selection & Tuning
# ─────────────────────────────────────────────────────────────────────
st.subheader("Model Selection")
col1, col2 = st.columns(2)

with col1:
    enable_tuning = st.checkbox(
        "Enable hyperparameter tuning",
        value=False,
        help="Search multiple hyperparameter combinations for the best model.",
    )

with col2:
    manual_model_override = st.checkbox(
        "Manual model override",
        value=False,
        help="Choose a specific model architecture instead of auto-detection.",
    )

manual_model_selection = None
if manual_model_override:
    if detected_type == "tabular":
        available_models = ["MLP-Small", "MLP-Medium", "MLP-Large"]
    elif detected_type == "image":
        available_models = ["Small-CNN", "MobileNetV2", "EfficientNetB0"]
    elif detected_type == "text":
        available_models = ["BiLSTM", "LSTM", "Text-CNN"]
    else:
        available_models = []
    if available_models:
        manual_model_selection = st.selectbox(
            "Select the model architecture:",
            available_models,
            help="Override auto-detection with a specific model.",
        )
    else:
        st.info("Upload a dataset first to see available models.")

# Tuning config
user_tuning_config = None
if enable_tuning:
    with st.expander("Tuning Settings"):
        max_trials = st.slider("Max trials", 2, 30, 8,
            help="Number of hyperparameter combinations to try.")
        epochs_per_trial = st.slider("Epochs per trial", 3, 30, 10)
        tune_lr = st.checkbox("Tune learning rate", value=True)
        tune_hidden = st.checkbox("Tune hidden layer sizes", value=True)
        tune_dropout = st.checkbox("Tune dropout rate", value=False)
        user_tuning_config = {
            "max_trials": max_trials,
            "epochs": epochs_per_trial,
            "tune_learning_rate": tune_lr,
            "tune_hidden_units": tune_hidden,
            "tune_dropout": tune_dropout,
        }

# ─────────────────────────────────────────────────────────────────────
# TRAIN BUTTON
# ─────────────────────────────────────────────────────────────────────
train_clicked = st.button("Train Model", type="primary", use_container_width=True)

if train_clicked:
    if train_file is None:
        st.error(
            "**No file uploaded.** "
            "Why: training requires a labelled dataset. "
            "What to do: upload a .csv, .xlsx, .txt, or .zip file above."
        )
    else:
        train_result = None
        try:
            # Validate file
            dataset_path = save_uploaded_file(train_file, prefix="train")
            validate_file_exists(dataset_path)

            # xlsx integrity check
            if dataset_path.lower().endswith(".xlsx"):
                import zipfile
                try:
                    with zipfile.ZipFile(dataset_path, "r") as z:
                        bad = z.testzip()
                        if bad is not None:
                            raise AutoDLInputError(
                                f"Corrupt Excel file entry: '{bad}'. "
                                "Why: the .xlsx file may have been partially downloaded or is damaged. "
                                "What to do: re-save the file and re-upload."
                            )
                except zipfile.BadZipFile:
                    raise AutoDLInputError(
                        f"The .xlsx file is not a valid ZIP archive. "
                        "Why: the file may be corrupted or in a different format disguised as .xlsx. "
                        "What to do: export as a proper .xlsx from Excel/LibreOffice and re-upload."
                    )

            # 2. Run training with st.status() progress
            with st.status(
                "Training model...",
                expanded=True,
                state="running",
            ) as training_status:
                st.write("Detecting dataset type...")
                time.sleep(0.2)

                st.write("Loading & validating data...")
                time.sleep(0.2)

                st.write("Detecting target column...")
                time.sleep(0.2)

                st.write("Running model search & CV...")
                train_result = train_pipeline(
                    dataset_path,
                    enable_tuning=enable_tuning,
                    tuning_config=user_tuning_config if enable_tuning else None,
                    manual_model_selection=manual_model_selection
                    if manual_model_override else None,
                    target_col=target_collected,
                )

                st.write("Computing metrics & generating report...")
                training_status.update(
                    label="Training complete!",
                    state="complete",
                    expanded=False,
                )

            # 3. Store in session
            st.session_state.train_result = train_result
            st.session_state.model_dir = train_result["model_dir"]
            st.session_state.model_id = train_result["model_id"]
            st.session_state.dataset_type = train_result["dataset_type"]

            # Register in central index (best-effort)
            try:
                register_model(train_result["model_dir"])
            except Exception:
                pass

            # 4. Show results
            st.success(f"**Training complete!** — Model ID: `{train_result['model_id']}`")
            st.markdown(f"**Dataset type:** `{train_result['dataset_type']}` | "
                        f"**Model:** `{train_result.get('model_name','?')}`")

            # ---- Model Comparison Table ----
            comparison = train_result.get("model_comparison", {})
            models = comparison.get("models", [])
            winner = comparison.get("selected", train_result.get("model_name", "Unknown"))
            reason = comparison.get("reason", "")

            if models:
                st.subheader("Model Comparison (CV Scores)")
                comp_df = pd.DataFrame(models)
                display_cols = ["name", "score", "stage", "description", "params"]
                display_cols = [c for c in display_cols if c in comp_df.columns]

                # Style: highlight winner
                def highlight_winner(row):
                    if row.get("name") == winner:
                        return ["background-color: #dbeafe; font-weight: bold"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    comp_df[display_cols].style.apply(highlight_winner, axis=1).format(
                        {"score": "{:.4f}"} if "score" in display_cols else {}
                    ),
                    use_container_width=True,
                )
                st.info(f"**Winner:** `{winner}` — {reason}")

            # ── Metrics ──
            metrics_path = os.path.join(train_result["model_dir"], "training_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                # Show summary metrics
                st.subheader("Validation Metrics")
                metric_cols = st.columns(4)
                for i, (key, val) in enumerate([
                    ("Accuracy", m.get("accuracy")),
                    ("Precision", m.get("precision")),
                    ("Recall", m.get("recall")),
                    ("F1 Score", m.get("f1_score")),
                ]):
                    metric_cols[i].metric(
                        label=key,
                        value=f"{val:.4f}" if isinstance(val, float) else str(val or "N/A"),
                    )

            # ── Confusion Matrix (Plotly) ──
            cm_data = m.get("confusion_matrix")
            if cm_data:
                st.subheader("Confusion Matrix")
                class_names = train_result.get("class_names")
                fig_cm = _build_confusion_matrix_fig(cm_data, class_names)
                st.plotly_chart(fig_cm, use_container_width=True)

            # ── Training Plots ──
            plots_dir = os.path.join(train_result["model_dir"], "plots")
            if os.path.isdir(plots_dir):
                st.subheader("Training Curves")
                cols = st.columns(2)
                loss_path = os.path.join(plots_dir, "loss.png")
                acc_path = os.path.join(plots_dir, "accuracy.png")
                cm_png_path = os.path.join(plots_dir, "confusion_matrix.png")
                if os.path.exists(loss_path):
                    cols[0].image(loss_path, caption="Loss Curve", use_container_width=True)
                if os.path.exists(acc_path):
                    cols[1].image(acc_path, caption="Accuracy Curve", use_container_width=True)
                if os.path.exists(cm_png_path):
                    st.image(cm_png_path, caption="Confusion Matrix", use_container_width=True)

            # ── Download Report Button ──
            report_file = None
            html_report = os.path.join(train_result.get("model_dir", ""), "report.html")
            pdf_report = train_result.get("report_path")
            if os.path.exists(html_report):
                report_file = html_report
                report_label = "Download HTML Report"
                report_mime = "text/html"
                report_name = f"AutoDL_report_{train_result['model_id']}.html"
            elif pdf_report and os.path.exists(pdf_report):
                report_file = pdf_report
                report_label = "Download PDF Report"
                report_mime = "application/pdf"
                report_name = f"AutoDL_report_{train_result['model_id']}.pdf"

            if report_file:
                st.subheader("Download Report")
                with open(report_file, "rb") as f:
                    st.download_button(
                        label=report_label,
                        data=f,
                        file_name=report_name,
                        mime=report_mime,
                        type="primary",
                    )

            # ── Classification Report ──
            cls_report_path = os.path.join(train_result.get("model_dir", ""), "classification_report.txt")
            if os.path.exists(cls_report_path):
                with st.expander("Classification Report (Detailed)"):
                    with open(cls_report_path) as f:
                        st.text(f.read())

        except AutoDLTargetAmbiguousError as e:
            st.error(f"**Target Column Required:** {str(e)}")
        except AutoDLInputError as e:
            st.error(f"**Data Error:** {str(e)}")
        except AutoDLTrainingError as e:
            st.error(f"**Training Error:** {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
        except FileNotFoundError as e:
            st.error(f"**File Error:** {str(e)}")
        except Exception as e:
            st.error(f"**Unexpected Error:** {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Predict using Trained Model
# ─────────────────────────────────────────────────────────────────────
st.header("Step 2 — Predict")
if st.session_state.model_dir:
    st.info(f"Using trained model: **`{st.session_state.model_id}`**")
    predict_file = st.file_uploader(
        "Upload unlabelled data (same type as training)",
        type=["csv", "xlsx", "txt", "zip"],
        key="predict_file",
    )

    if st.button("Run Prediction"):
        if predict_file is None:
            st.error(
                "**No prediction file uploaded.** "
                "Why: prediction requires unlabelled data. "
                "What to do: upload a file with the same columns/structure as the training data."
            )
        else:
            try:
                dataset_path = save_uploaded_file(predict_file, prefix="predict")
                validate_file_exists(dataset_path)

                with st.spinner("Running prediction..."):
                    pred_result = predict_pipeline(st.session_state.model_dir, dataset_path)

                preds = pred_result.get("predictions", [])
                classes = pred_result.get("classes", [])

                # Build prediction dataframe
                df_pred = pd.DataFrame({
                    "Index": list(range(len(preds))),
                    "Predicted_Class": preds,
                })
                if classes:
                    df_pred["Class_Name"] = df_pred["Predicted_Class"].apply(
                        lambda i: classes[int(i)] if 0 <= int(i) < len(classes) else str(i)
                    )

                st.subheader("Predictions (first 50 rows)")
                st.dataframe(df_pred.head(50), use_container_width=True)

                # Download CSV
                st.download_button(
                    "Download Predictions CSV",
                    data=df_pred.to_csv(index=False).encode("utf-8"),
                    file_name=f"AutoDL_predictions_{st.session_state.model_id}.csv",
                )

                # Prediction report
                report_path = pred_result.get("report_path")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "Download Prediction Report (PDF)",
                            data=f,
                            file_name=f"AutoDL_predict_{st.session_state.model_id}.pdf",
                        )

            except AutoDLInputError as e:
                st.error(f"**Prediction Error:** {str(e)}")
            except Exception as e:
                st.error(f"**Prediction Error:** {str(e)}")
                with st.expander("Technical Details"):
                    st.code(traceback.format_exc())
else:
    st.info("Train a model first to unlock prediction mode.")

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AutoDL — Automated Classification with transparent validation | "
    f"Seed: {RANDOM_SEED} | Model Registry: `{MODEL_REGISTRY_DIR}`"
)