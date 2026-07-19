# AutoDL

AutoDL is an MVP for non-technical users who need to turn a labelled dataset into a reproducible classification model, a held-out validation report, and batch predictions. It currently supports tabular, image, and text classification.

## What it does today

1. Accepts labelled tabular (`.csv`, `.xlsx`), image (`.zip`), or text (`.txt`) data.
2. Validates and preprocesses the input, then selects a transparent baseline architecture or honours a manual choice.
3. Trains the model, evaluates it on a held-out validation split, saves the model and preprocessing artifacts, and creates a PDF report.
4. Accepts compatible unlabelled data for batch predictions.

The application is classification-only. It does **not** currently support regression, multi-label classification, time series, object detection, semantic segmentation, or production-grade AutoML model comparison.

## Repository layout

```text
AutoDL/
├── app.py                  # Streamlit interface
├── src/                    # Application package (renamed from explainDL)
│   ├── core/               # Training and prediction orchestration
│   ├── data/               # Input readers and type detection
│   ├── preprocessing/      # Modality-specific transformations
│   ├── model_selection/    # Baseline architectures and tuning
│   ├── training/           # Fit loop and held-out metrics
│   ├── explainability/     # Reports and explainability helpers
│   └── registry/           # Local model index
├── pyproject.toml          # Primary dependency declaration
└── uv.lock                 # Reproducible dependency lock
```

All Python imports use `src`, for example `from src.core.pipeline_train import train_pipeline`. The old `explainDL/` directory has been removed.

## Run locally

Requires Python 3.11 or newer and a supported TensorFlow environment.

```bash
uv sync
uv run streamlit run app.py
```

Alternatively, install the locked requirements and run `streamlit run app.py`.

## Dataset contracts

| Modality | Training input | Prediction input | Important contract |
| --- | --- | --- | --- |
| Tabular | CSV/XLSX, at least 10 rows, features plus target | CSV/XLSX with the original feature columns | The target is currently assumed to be the final training column. |
| Images | ZIP with at least two class-name folders and 10 valid images | ZIP of images; folders optional | Each class needs enough examples for a validation split. |
| Text | TXT, one `label<TAB>text` or `label,text` record per line, at least 10 lines | TXT, one text per line | Classification labels must have at least two distinct values. |

Uploaded data and generated model artifacts are stored locally. The local registry is ignored by Git.

## Validation and explainability

Reported accuracy, precision, recall, F1, confusion matrix, and classification report are calculated from held-out validation data—not the training examples. The selected model’s explanation distinguishes a scale-based baseline recommendation from an actual accuracy comparison.

The PDF reports currently provide metrics, learning curves, confusion matrices, class distributions, and plain-language summaries. SHAP, LIME, and Grad-CAM modules exist as utilities, but they are not yet consistently produced for every trained model; do not represent their availability as a complete explainability workflow.

## Known MVP limitations

- The model selector chooses a deterministic baseline from modality, feature shape, and sample count. It does not run a fair cross-validated model tournament.
- The target-column assumption is unsuitable for many datasets; a production UI should make it explicit.
- Training runs synchronously inside Streamlit, so it is not suitable for long jobs, multiple users, or GPU scheduling.
- Artifacts lack dataset versioning, lineage, approval workflows, monitoring, authentication, and a deployment API.
- TensorFlow/Keras models are saved locally; there is no model promotion, rollback, or serving layer.

## Production direction

The next implementation should split the product into a browser UI, a job API/queue, object storage, a metadata database, and a model-serving service. Add schema/target selection, data-profiling and quality gates, stratified cross-validation, leakage checks, reproducible experiment tracking, calibrated probabilities, per-prediction explanations, drift monitoring, RBAC, and audit logs before presenting it as a production decision-support system.

See the project review delivered with this change for a prioritized roadmap.
