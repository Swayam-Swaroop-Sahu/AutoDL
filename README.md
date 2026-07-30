# AutoDL — Phase 1: Correctness

AutoDL is an automated classification engine for non-technical users. It accepts labelled tabular, image, or text data and produces a reproducible model, a transparent validation report, and batch predictions. Phase 1 introduces evidence-based target detection, real cross-validated model comparison, unified multiclass code paths, smart binary thresholds, and a reliability contract (timeout + fallback) that prevents the pipeline from ever hanging.

## What it does today

1. **Detect target** — automatically scores every column with a Target Likelihood Score (name + cardinality + dtype + predictability) and either auto-selects, escalates to the user, or rejects when no classification target is present.
2. **Compare models** — successive-halving search across candidate families with 3-fold stratified CV. Per-candidate scores are recorded in metadata so reports are evidence-based.
3. **Train + evaluate** — unified multiclass code path (single softmax head, `sparse_categorical_crossentropy`, `argmax` decode); smart binary threshold via Youden's J on the validation set.
4. **Predict** — applies the trained model to new data; the saved `LabelEncoder` and (for binary) the optimized threshold round-trip back to the original class labels.
5. **Reliability** — every stage (load, preprocess, search, train) runs under a wall-clock timeout with a deterministic fallback. The pipeline never hangs.

The application is classification-only. It does **not** currently support regression, multi-label classification, time series, object detection, semantic segmentation, or production-grade AutoML hyperparameter optimization.

## Reliability

Every pipeline stage is wrapped by a circuit breaker (`src/core/circuit_breaker.py`) that enforces a deadline and a structured fallback:

- `load` — raises immediately on failure (no fallback; bad input is the user's fault).
- `target_detect` — raises on `human_required` / `not_classification` (caller must confirm or correct).
- `search` — falls back to a single `LogisticRegression` baseline (logged as `TIMEOUT_FALLBACK: search`).
- `train` — falls back to zeroed metrics so the pipeline still produces an artifact.
- `evaluate` / `report` — best-effort; logged warnings, never fatal.

The full sequence is appended to `circuit_breaker.jsonl` inside each model directory for post-mortem inspection.

## Repository layout

```text
AutoDL/
├── app.py                  # Streamlit interface
├── src/
│   ├── core/               # Training + prediction orchestration, config, circuit breaker
│   ├── data/               # Input readers (CSV/XLSX, ZIP, TXT) and type detection
│   ├── preprocessing/      # Modality-specific transformers + leakage detection
│   ├── model_selection/    # Hand-rolled successive-halving + per-modality candidates
│   ├── target_detection/   # TLS scoring + escalation (resolve_target)
│   ├── training/           # Fit loop, threshold optimization, metrics
│   ├── explainability/     # Reports and explainability helpers
│   └── registry/           # Local model index (file-locked RMW)
├── tests/
│   ├── test_target_detection.py
│   ├── test_search.py
│   ├── test_multiclass.py
│   ├── test_threshold.py
│   ├── test_circuit_breaker.py
│   ├── test_bugfix_phase1e.py
│   └── integration/        # End-to-end on messy real-world-shaped data
├── PHASE_1_SUMMARY.md      # Detailed change log
├── pyproject.toml          # Primary dependency declaration
└── uv.lock                 # Reproducible dependency lock
```

All Python imports use `src`, for example `from src.core.pipeline_train import train_pipeline`.

## Run locally

Requires Python 3.11 or newer and a supported TensorFlow environment.

```bash
uv sync
uv run streamlit run app.py
```

Alternatively, install the locked requirements and run `streamlit run app.py`.

## Model Selection

AutoDL uses a hand-rolled successive-halving model search (`src/model_selection/search.py`):

- 3 stages: fidelity 0.1 (25% time budget), 0.5 (35% time budget), 1.0 (40% time budget).
- 3-fold stratified CV per trial at each fidelity.
- Top 50% promoted each stage.
- Stage-timeout enforcement: if a stage's time budget is exceeded, no further trials start that stage.
- Crash resilience: a candidate that throws is caught and scored 0.0; the search continues.

Per-modality candidates:

| Modality | Candidates |
| --- | --- |
| Tabular | GradientBoostingClassifier, RandomForestClassifier, LogisticRegression |
| Text | TF-IDF + LinearSVC (calibrated), TF-IDF + LogisticRegression, TF-IDF + ComplementNB |
| Image | MobileNetV2 (<500 samples), ResNet50 (500–5k), EfficientNetB0 (>5k) — frozen backbone features + LR head |

## Multiclass Support

There is one code path for binary, 3-class, 5-class, and K-class classification:

- Keras head: `Dense(units=num_classes, activation="softmax")` with `loss="sparse_categorical_crossentropy"`.
- sklearn: handles multiclass natively; `predict_proba` always has shape `(N, num_classes)`.
- Decode: `np.argmax(preds_proba, axis=1)`.

For binary classification, the trained model still emits a `(N, 2)` softmax output. The pipeline additionally computes a threshold via Youden's J statistic on the validation set and stores it in `meta.json["binary_threshold"]`. At predict time, that threshold is applied to `preds_proba[:, 1]` before argmax, so imbalanced binary problems no longer default to 0.5.

## Dataset contracts

| Modality | Training input | Prediction input | Important contract |
| --- | --- | --- | --- |
| Tabular | CSV/XLSX, at least 10 rows, features plus target | CSV/XLSX with the original feature columns | The target is detected automatically (TLS scoring). You may also pass an explicit `target_col=` to bypass detection. |
| Images | ZIP with at least two class-name folders and 10 valid images | ZIP of images; folders optional | Each class needs enough examples for a validation split. Corrupted images are skipped without crashing. |
| Text | TXT, one `label<TAB>text` or `label,text` record per line, at least 10 lines | TXT, one text per line | Classification labels must have at least two distinct values. Non-ASCII text (Chinese, Arabic, Cyrillic) is preserved. |

Uploaded data and generated model artifacts are stored locally. The local registry is ignored by Git.

## Validation and explainability

Reported accuracy, precision, recall, F1, confusion matrix, and classification report are calculated from held-out validation data—not the training examples. The selected model's explanation distinguishes a CV-measured model comparison (search winner) from a recommendation flag. Per-candidate CV scores from the search are stored in `meta.json["search_results"]` and rendered in the PDF report.

## Known limitations

- LightGBM is unavailable on this host (Application Control policy); the tabular candidate factory uses `GradientBoostingClassifier` from sklearn instead.
- The fine-tuned DistilBERT text candidate from the original plan is deferred (offline environment, no network for HF downloads).
- `keras_tuner` is no longer wired into the pipeline; the Streamlit UI's "Enable tuning" checkbox is preserved for backward compatibility but currently has no effect.
- Training runs synchronously inside Streamlit, so it is not suitable for long jobs, multiple users, or GPU scheduling.

## Production direction

The next implementation should split the product into a browser UI, a job API/queue, object storage, a metadata database, and a model-serving service. Add data-profiling and quality gates, calibrated probabilities, per-prediction explanations, drift monitoring, RBAC, and audit logs before presenting it as a production decision-support system.
