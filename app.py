# app.py (root of project) — Streamlit UI for ExplainDL
import streamlit as st
import os
import tempfile
import pandas as pd
import traceback
import json

from explainDL.core.pipeline_train import train_pipeline
from explainDL.core.pipeline_predict import predict_pipeline
from explainDL.registry import register_model
from explainDL.data.detect_type import detect_dataset_type

# -----------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="ExplainDL", page_icon="🤖", layout="wide")
st.title("ExplainDL – Automated Deep Learning Tool")
st.write("Upload a labelled dataset to train a model, then upload unlabelled data for prediction.")

# -----------------------------------------------------------
# HELPER: save uploaded file safely (avoids NamedTemporaryFile on Windows)
# -----------------------------------------------------------
def save_uploaded_file(uploaded_file, prefix="explaindl"):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    ext = os.path.splitext(uploaded_file.name)[1]
    safe_name = uploaded_file.name.replace(" ", "_").replace(":", "_")
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, f"{prefix}_{safe_name}")
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "model_dir" not in st.session_state:
    st.session_state.model_dir = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "dataset_type" not in st.session_state:
    st.session_state.dataset_type = None

# -----------------------------------------------------------
# TRAIN SECTION
# -----------------------------------------------------------
st.header("Step 1 — Train Model")
train_file = st.file_uploader("Upload LABELLED dataset (CSV, XLSX, TXT, ZIP)", type=["csv", "xlsx", "txt", "zip"])

# preview small tabular
if train_file and train_file.name.lower().endswith((".csv", ".xlsx")):
    try:
        train_file.seek(0)
        df_preview = pd.read_csv(train_file, nrows=5) if train_file.name.lower().endswith(".csv") else pd.read_excel(train_file, nrows=5)
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_preview)
    except Exception as e:
        st.warning(f"Preview unavailable: {e}")

# Model Selection Mode
st.subheader("Model Selection")
model_selection_mode = st.radio(
    "Model Selection Mode",
    ["Automatic (Recommended)", "Manual Override"],
    help="Choose automatic selection or manually select a model"
)

manual_model_selection = None
if model_selection_mode == "Manual Override":
    # We'll populate this after detecting dataset type
    st.info("Please upload a dataset first to see available models.")

enable_tuning = st.checkbox("Enable hyperparameter tuning", value=False, help="If enabled, ExplainDL will try tuning (best-effort).")
if enable_tuning:
    st.subheader("Hyperparameter Tuning Settings")

    tuning_mode = st.selectbox(
        "Tuning Mode",
        ["Fast (max_trials = 4)", "Balanced (max_trials = 8)", "Thorough (max_trials = 15)"],
        help="Choose how extensive the tuning search should be."
    )

    if tuning_mode == "Fast (max_trials = 4)":
        max_trials = 4
        epochs = 5
    elif tuning_mode == "Balanced (max_trials = 8)":
        max_trials = 8
        epochs = 10
    else:  # Thorough
        max_trials = 15
        epochs = 15

    # DISPLAY PARAMETERS FOR USER CONTROL
    st.markdown("### Manual Overrides (optional)")

    max_trials_user = st.number_input(
        "Max trials",
        value=max_trials,
        min_value=2,
        max_value=50,
        step=1,
        help="Number of hyperparameter combinations to try."
    )

    epochs_user = st.number_input(
        "Training epochs per trial",
        value=epochs,
        min_value=3,
        max_value=50,
        step=1
    )

    tune_learning_rate = st.checkbox("Tune learning rate", value=True)
    tune_hidden_units = st.checkbox("Tune hidden layer sizes", value=True)
    tune_dropout = st.checkbox("Tune dropout rate", value=False)

    # pack user settings
    user_tuning_config = {
        "max_trials": int(max_trials_user),
        "epochs": int(epochs_user),
        "tune_learning_rate": tune_learning_rate,
        "tune_hidden_units": tune_hidden_units,
        "tune_dropout": tune_dropout,
    }

else:
    user_tuning_config = None

# Detect dataset type and show available models for manual selection
if train_file and model_selection_mode == "Manual Override":
    try:
        dataset_path_temp = save_uploaded_file(train_file, prefix="temp_detect")
        dataset_type = detect_dataset_type(dataset_path_temp)
        
        available_models = []
        if dataset_type == "tabular":
            available_models = ["MLP-Small", "MLP-Medium", "MLP-Large"]
        elif dataset_type == "image":
            available_models = ["Small-CNN", "MobileNetV2", "EfficientNetB0"]
        elif dataset_type == "text":
            available_models = ["BiLSTM", "LSTM", "Text-CNN"]
        else:
            st.warning(f"Dataset type '{dataset_type}' detected. Manual model selection may not be available.")
        
        if available_models:
            manual_model_selection = st.selectbox(
                "Select Model",
                available_models,
                help="Choose a specific model architecture to use",
                key="manual_model_select"
            )
            st.info(f"**Detected dataset type:** {dataset_type}")
        else:
            manual_model_selection = None
    except Exception as e:
        st.warning(f"Could not detect dataset type: {e}")
        manual_model_selection = None
elif model_selection_mode == "Automatic (Recommended)":
    manual_model_selection = None

train_clicked = st.button("Train Model")
train_result = None

if train_clicked:
    if train_file is None:
        st.error("Please upload a labelled dataset.")
    elif model_selection_mode == "Manual Override" and manual_model_selection is None:
        st.error("Please select a model in Manual Override mode.")
    else:
        try:
            dataset_path = save_uploaded_file(train_file, prefix="train")

            # integrity check for xlsx (avoid corrupted zip read errors)
            if dataset_path.lower().endswith(".xlsx"):
                import zipfile
                try:
                    with zipfile.ZipFile(dataset_path, "r") as z:
                        bad = z.testzip()
                        if bad is not None:
                            raise Exception(f"Corrupt xlsx entry: {bad}")
                except Exception as e:
                    st.error("Uploaded Excel file appears corrupted or unreadable.")
                    st.code(str(e))
                    raise

            with st.spinner("Training model... (this can take some time)"):
                train_result = train_pipeline(
                    dataset_path, 
                    enable_tuning=enable_tuning,
                    tuning_config=user_tuning_config if enable_tuning else None,
                    manual_model_selection=manual_model_selection if model_selection_mode == "Manual Override" else None
                )

            # save in session
            st.session_state.model_dir = train_result["model_dir"]
            st.session_state.model_id = train_result["model_id"]
            st.session_state.dataset_type = train_result["dataset_type"]

            # register in central index (best-effort)
            try:
                register_model(train_result["model_dir"])
            except Exception:
                pass

            st.success("Training completed.")
            st.write("**Model ID:**", f"`{train_result['model_id']}`")
            st.write("**Dataset type:**", f"`{train_result['dataset_type']}`")
            if train_result.get("class_names"):
                st.write("Class names:", train_result["class_names"])
            
            # Display Model Selection Information
            if train_result.get("model_comparison"):
                comparison = train_result["model_comparison"]
                st.subheader("📊 Model Selection Details")
                
                # Selected model info
                selected_name = comparison.get("selected", train_result.get("model_name", "Unknown"))
                reason = comparison.get("reason", "No reason provided")
                
                st.markdown(f"**✅ Selected Model:** `{selected_name}`")
                st.info(f"**Reason:** {reason}")

                # Optional detailed selection explanation
                selection_exp_path = train_result.get("selection_explanation_path")
                if selection_exp_path and os.path.exists(selection_exp_path):
                    with open(selection_exp_path, "r", encoding="utf-8") as f:
                        st.markdown("**How the model was chosen (text summary):**")
                        st.text(f.read())
                
                # Model comparison table
                if comparison.get("models"):
                    st.markdown("### Model Comparison")
                    comparison_df = pd.DataFrame(comparison["models"])
                    
                    # Format the comparison table
                    display_df = pd.DataFrame({
                        "Model": comparison_df["name"],
                        "Score": comparison_df["score"].apply(lambda x: f"{x:.4f}"),
                        "Description": comparison_df["description"],
                        "Parameters": comparison_df["params"],
                        "Pros": comparison_df["pros"],
                        "Cons": comparison_df["cons"]
                    })
                    
                    # Display table with selected model marked
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show which model was selected
                    st.markdown(f"**Note:** The selected model ({selected_name}) is highlighted in the comparison above.")
                    
                    # Show score comparison chart
                    st.markdown("### Score Comparison")
                    score_df = pd.DataFrame({
                        "Model": comparison_df["name"],
                        "Score": comparison_df["score"]
                    })
                    st.bar_chart(score_df.set_index("Model"))

            # Training explanation text
            training_exp_path = train_result.get("training_explanation_path")
            if training_exp_path and os.path.exists(training_exp_path):
                st.subheader("📝 Training Explanation")
                with open(training_exp_path, "r", encoding="utf-8") as f:
                    st.text(f.read())

            # show visual explainability (plots live)
            plots_dir = os.path.join(train_result["model_dir"], "plots")
            if os.path.isdir(plots_dir):
                st.subheader("Training — Plots & Visuals")
                cols = st.columns(2)
                # loss
                loss_path = os.path.join(plots_dir, "loss.png")
                acc_path = os.path.join(plots_dir, "accuracy.png")
                cm_path = os.path.join(plots_dir, "confusion_matrix.png")
                if os.path.exists(loss_path):
                    cols[0].image(loss_path, caption="Loss Curve", use_container_width=True)
                if os.path.exists(acc_path):
                    cols[1].image(acc_path, caption="Accuracy Curve", use_container_width=True)
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)

            # show metrics JSON & classification report if present
            metrics_json = os.path.join(train_result["model_dir"], "training_metrics.json")
            cls_report = os.path.join(train_result["model_dir"], "classification_report.txt")
            if os.path.exists(metrics_json):
                st.subheader("Training Metrics (summary)")
                try:
                    with open(metrics_json, "r") as f:
                        m = json.load(f)
                    # remove verbose arrays if present
                    m_short = {k: v for k, v in m.items() if k not in ("y_true", "y_pred")}
                    st.json(m_short)
                except Exception:
                    st.text("Could not load metrics JSON.")
            if os.path.exists(cls_report):
                st.subheader("Classification Report (text)")
                with open(cls_report) as f:
                    st.text(f.read())

            # download report pdf
            if train_result.get("report_path") and os.path.exists(train_result["report_path"]):
                with open(train_result["report_path"], "rb") as f:
                    st.download_button("Download Training Report (PDF)", data=f, file_name=f"ExplainDL_train_{train_result['model_id']}.pdf")

        except Exception:
            st.error("Error during training — see trace below.")
            st.code(traceback.format_exc())

# -----------------------------------------------------------
# PREDICTION SECTION
# -----------------------------------------------------------
st.header("Step 2 — Predict using Trained Model")
if st.session_state.model_dir:
    st.info(f"Using model: `{st.session_state.model_id}`")
    predict_file = st.file_uploader("Upload UNLABELLED dataset (same type as training)", type=["csv", "xlsx", "txt", "zip"], key="predict_file")

    if st.button("Run Prediction"):
        if predict_file is None:
            st.error("Please upload a dataset for prediction.")
        else:
            try:
                dataset_path = save_uploaded_file(predict_file, prefix="predict")
                # optional xlsx integrity check
                if dataset_path.lower().endswith(".xlsx"):
                    import zipfile
                    try:
                        with zipfile.ZipFile(dataset_path, "r") as z:
                            bad = z.testzip()
                            if bad is not None:
                                raise Exception(f"Corrupt xlsx entry: {bad}")
                    except Exception as e:
                        st.error("Uploaded Excel file appears corrupted or unreadable.")
                        st.code(str(e))
                        raise

                with st.spinner("Running prediction..."):
                    pred_result = predict_pipeline(st.session_state.model_dir, dataset_path)

                preds = pred_result.get("predictions")
                classes = pred_result.get("classes")
                report_path = pred_result.get("report_path")

                # Dataframe of predictions
                df_pred = pd.DataFrame({"Index": list(range(len(preds))), "Predicted": preds})
                if classes:
                    df_pred["Class_Name"] = df_pred["Predicted"].apply(lambda i: classes[int(i)] if 0 <= int(i) < len(classes) else str(i))

                st.subheader("Predictions (first 50 rows)")
                st.dataframe(df_pred.head(50))

                st.download_button("Download Predictions CSV", data=df_pred.to_csv(index=False).encode("utf-8"),
                                   file_name=f"ExplainDL_predictions_{st.session_state.model_id}.csv")

                # show prediction plots & explanation
                plots_dir = os.path.join(st.session_state.model_dir, "plots")
                if os.path.isdir(plots_dir):
                    st.subheader("Prediction — Visuals")
                    hist_path = os.path.join(plots_dir, "prediction_hist.png")
                    pie_path = os.path.join(plots_dir, "prediction_pie.png")
                    cols = st.columns(2)
                    if os.path.exists(hist_path):
                        cols[0].image(hist_path, caption="Prediction Histogram", use_container_width=True)
                    if os.path.exists(pie_path):
                        cols[1].image(pie_path, caption="Prediction Distribution (pie)", use_container_width=True)

                # prediction explanation text
                explanation_txt = os.path.join(st.session_state.model_dir, "prediction_explanation.txt")
                if os.path.exists(explanation_txt):
                    st.subheader("Simple explanation (non-technical)")
                    with open(explanation_txt) as f:
                        st.text(f.read())

                # download prediction pdf
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("Download Prediction Report (PDF)", data=f,
                                           file_name=f"ExplainDL_predict_{st.session_state.model_id}.pdf")

            except Exception:
                st.error("Error during prediction — see trace below.")
                st.code(traceback.format_exc())

else:
    st.info("Train a model first to unlock prediction mode.")

st.markdown("---")
st.caption("© 2025 ExplainDL | Developed by Swayam Swaroop Sahu")
