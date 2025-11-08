import streamlit as st
from explainDL.pipeline import run_pipeline
import pandas as pd
import io
import os
import tempfile

# -----------------------------------------------
# Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="ExplainDL - Automated Deep Learning Analysis",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------------------------
# Title and Intro
# -----------------------------------------------
st.title("ExplainDL: Automated Deep Learning Analysis Tool")
st.markdown("""
Welcome to **ExplainDL**, an intelligent Deep Learning automation system that:
- Detects your dataset type (tabular, image, or text)
- Automatically preprocesses and trains deep learning models
- Selects the best-performing model
- Provides explainability insights and performance reports

Simply upload your dataset and let ExplainDL handle the rest!
""")

# -----------------------------------------------
# File Upload
# -----------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset file (CSV, XLSX, ZIP for images, or TXT)",
    type=["csv", "xlsx", "zip", "txt"]
)

# -----------------------------------------------
# Sidebar Options
# -----------------------------------------------
st.sidebar.header("Configuration")
auto_mode = st.sidebar.toggle("Enable Auto Mode (recommended)", True)
show_explainability = st.sidebar.toggle("Show Explainability Report", True)
enable_tuning = st.sidebar.toggle("Enable Auto Hyperparameter Tuning", False)
max_trials = st.sidebar.slider("Tuning Trials (if enabled)", 5, 50, 10)

# -----------------------------------------------
# Run Pipeline
# -----------------------------------------------
if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Try reading the first few rows if tabular
    if uploaded_file.name.endswith(('.csv', '.xlsx')):
        try:
            uploaded_file.seek(0)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.subheader("Sample of Uploaded Data")
            st.dataframe(df.head())
        except Exception as e:
            st.warning(f"Could not preview file: {e}")

    st.markdown("---")
    st.subheader("Run Automated Deep Learning Pipeline")

    if st.button("Start Analysis"):
        with st.spinner("Running the automated pipeline... This may take a few minutes ⏳"):
            try:
                # Save uploaded file to a temporary directory before passing it
                temp_dir = tempfile.mkdtemp(prefix="ExplainDL_")
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                uploaded_file.seek(0)  # Reset pointer
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Run pipeline on the saved file path
                results = run_pipeline(
                    temp_path,
                    auto_mode=auto_mode,
                    show_explainability=show_explainability,
                    enable_tuning=enable_tuning,
                    max_trials=max_trials
                )

                # Display Metrics
                st.success("Pipeline completed successfully!")
                if "metrics" in results and not results["metrics"].empty:
                    st.subheader("Model Performance Summary")
                    st.dataframe(results["metrics"])

                # Display Best Model Explainability
                if show_explainability and "explainability_fig" in results and results["explainability_fig"]:
                    st.subheader("Model Explainability Insights")
                    st.image(results["explainability_fig"], use_container_width=True)

                # Download Option
                if "report_path" in results and results["report_path"]:
                    with open(results["report_path"], "rb") as file:
                        st.download_button(
                            label="Download Analysis Report",
                            data=file,
                            file_name="ExplainDL_Report.pdf"
                        )

            except Exception as e:
                st.error(f"An error occurred while running the pipeline: {e}")
else:
    st.info("Please upload a dataset file to begin.")

# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.caption("© 2025 ExplainDL | Developed by Swayam Swaroop Sahu")
