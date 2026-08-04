# AutoDL - Multimodal AutoML for Classification

> **Automated target detection, model selection, training, and explainable reporting for tabular, image, and text classification - all local, no code required.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-72%20passing-brightgreen)](tests/)

---

## Overview

AutoDL is a **local-first AutoML tool** for supervised classification. It handles the full pipeline:

1. **Target detection** - Ranks candidate columns via name + cardinality heuristics (TLS); user confirms selection (never auto-picks silently)
2. **Model selection** - 3-stage successive-halving search with 3-fold stratified CV; bounded time budget; crash-resilient
3. **Training** - Unified multiclass path (softmax + sparse_categorical_crossentropy); Youden's J threshold optimization for binary
4. **Reliability** - Circuit breaker: every stage has wall-clock timeout + deterministic fallback; pipeline never hangs
5. **Explainability** - Permutation importance (tabular), narrative summary, HTML/PDF report, confusion matrix, training curves
6. **Interfaces** - Streamlit UI (drag-and-drop, live progress, charts) + CLI (`auto.py train|predict`) + optional FastAPI `/predict`

**Supported modalities:** Tabular (CSV/XLSX), Image (ZIP folders), Text (labelled TXT)

---

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`

### Installation
```bash
# Clone and install
git clone <repo-url> AutoDL
cd AutoDL
uv sync          # or: pip install -e .
```

### Streamlit UI (Recommended)
```bash
uv run streamlit run app.py
```
Open `http://localhost:8501` → Upload labelled data → Confirm target → Click **Train Model**.

### CLI
```bash
# Train with explicit target
uv run python auto.py train --data data.csv --target survived

# Train with interactive target selection
uv run python auto.py train --data data.csv

# Predict on new data
uv run python auto.py predict --data test.csv --model <model_id>
```

### FastAPI Serving (Optional)
```bash
uv run python auto.py serve --model <model_id> --port 8000
# POST /predict with JSON or multipart form
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Target Likelihood Scoring (TLS)** | Ranks columns by name patterns (`target`, `label`, `is_`, `churn`...) + cardinality signals (binary=1.0, low-card=0.8, ID=0.0). User *must* confirm. |
| **Successive-Halving Search** | 3 fidelities (10%/50%/100% data), 3-fold stratified CV, 25%/35%/40% time split. Top 50% promoted per stage. |
| **Unified Multiclass** | Single code path: `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy` for 2, 3, 5, K classes. |
| **Youden's J Threshold** | Optimizes `TPR - FPR` on validation set for binary; stored in metadata; applied at prediction time. |
| **Circuit Breaker Reliability** | Every stage: timeout + fallback + structured logging. Pipeline cannot hang. |
| **Permutation Importance** | `sklearn.inspection.permutation_importance` (tabular); graceful fallback for unsupported models. |
| **Narrative Summary** | Auto-generated plain-English: *"Model achieves 87% accuracy. Relies most on 'age'. No quality issues detected."* |
| **HTML/PDF Reports** | Confusion matrix, training curves, model comparison, feature importance, quality warnings, narrative. |
| **Quality Warnings** | Detects class imbalance, missing values, ID columns, low sample count, leakage risk. |

---

## Models Used

AutoDL combines **deep learning** and **classical ML** models per modality. The successive-halving search compares all of them with cross-validation and picks the best-fit winner.

### Tabular (CSV / XLSX)

**Deep Learning (MLPs):**
- `MLP-Small` - 1 hidden layer (64 units, ReLU), early stopping
- `MLP-Medium` - 2 hidden layers (128->64, ReLU), L2 regularization
- `MLP-Deep` - 3 hidden layers (256->128->64, ReLU), low L2

**Classical ML:**
- `GradientBoostingClassifier` - gradient-boosted trees (sklearn)
- `RandomForestClassifier` - bagged decision trees
- `LogisticRegression` - linear baseline with L2

### Images (ZIP)

**Deep Learning:**
- `Small-CNN` - 3-conv-layer CNN trained from scratch (32->64->128 filters, GAP)
- `MobileNetV2` - transfer learning (frozen backbone + LR head, ~3.5M params)
- `ResNet50` - transfer learning (frozen backbone + LR head, ~25M params)
- `EfficientNetB0` - transfer learning (frozen backbone + LR head, ~5M params)

*Note:* Heads on top of frozen feature extractors use Logistic Regression. There is no pure-classical image path; images rely on neural feature extractors.

### Text (TXT)

**Deep Learning (Keras):**
- `BiLSTM` - bidirectional LSTM (64 units) + dense head
- `LSTM` - single-layer LSTM (64 units) + dense head
- `Text-CNN` - 1D conv (128 filters, kernel=5) + global max pool + dense head

**Classical ML (TF-IDF + sklearn):**
- `TFIDF_LinearSVM` - TF-IDF (uni+bi-grams) + Linear SVM (calibrated)
- `TFIDF_LogReg` - TF-IDF + L2 logistic regression
- `TFIDF_ComplementNB` - TF-IDF + Complement Naive Bayes (imbalanced-friendly)

---

## Supported Data Formats

| Modality | Training Input | Prediction Input | Requirements |
|----------|----------------|------------------|--------------|
| **Tabular** | CSV / XLSX | CSV / XLSX (same features) | ≥10 rows, target column present |
| **Image** | ZIP with class folders | ZIP (folders optional) | ≥2 classes, ≥10 valid images/class |
| **Text** | TXT (`label<TAB>text` per line) | TXT (one text per line) | ≥10 lines, ≥2 distinct labels |

---

## Architecture

```
AutoDL/
├── app.py                 # Streamlit UI
├── auto.py                # CLI entry point (train | predict | serve)
├── src/
│   ├── core/              # Pipeline orchestration, circuit breaker, config, validation
│   ├── data/              # Loaders + type detection (tabular, image, text)
│   ├── preprocessing/     # Modality-specific transformers + leakage detection
│   ├── model_selection/   # Successive-halving search + per-modality candidates
│   ├── target_detection/  # TLS scoring (rank_target_candidates)
│   ├── training/          # Fit loop, Youden's J threshold, metrics
│   ├── explainability/    # Permutation importance, narrative, SHAP (opt-in), reports
│   ├── quality/           # Data quality summarization
│   ├── registry/          # Local model index (file-locked)
│   └── utils/             # Logger, helpers
├── tests/
│   ├── unit/              # Core component tests
│   └── integration/       # End-to-end on messy real-world data
├── model_registry/        # Trained models (gitignored)
├── pyproject.toml         # Dependencies
├── uv.lock                # Reproducible lock file
└── README.md              # This file
```

---

## How It Works

### 1. Target Detection (Tabular Only)
AutoDL computes a **Target Likelihood Score (TLS)** for every column:
- **Name signal (50%)** - Matches positive patterns (`arget`, `label`, `is_`, `churn`, `survived`...) and penalizes negative patterns (`id`, `uuid`, `imestamp`, `email`...)
- **Cardinality signal (50%)** - Binary=1.0, low-card categorical=0.8, high-card continuous=0.1, unique-per-row=0.0

The ranked table is displayed; **user confirmation is mandatory** - no silent auto-selection.

### 2. Model Selection
**Successive-halving** with 3 stages:
| Stage | Data Fidelity | Time Budget | Promotion |
|-------|---------------|-------------|-----------|
| 1 | 10% | 25% (150s) | Top 50% |
| 2 | 50% | 35% (210s) | Top 50% |
| 3 | 100% | 40% (240s) | Winner |

- 3-fold stratified CV per trial
- Metric: **balanced_accuracy** (macro recall - handles imbalance)
- Crash resilience: failed candidates score 0.0; search continues
- Timeout enforcement: stage budget exceeded → stop new trials, promote survivors

### 3. Training & Thresholds
- **Unified head**: `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy`
- **Binary**: Youden's J (`max(TPR - FPR)`) on validation set → threshold stored in `meta.json` → applied at prediction
- **Multiclass**: `argmax(proba)` directly

### 4. Reliability (Circuit Breaker)
Every pipeline stage wrapped with:
- Thread-based deadline (timeout)
- Structured logging (success/timeout/exception/fallback)
- Deterministic fallback callable
- JSONL checkpoint (`circuit_breaker.jsonl`)

| Stage | Timeout | Fallback |
|-------|---------|----------|
| `load` | 120s | None (re-raises) |
| `preprocess` | 180s | None (re-raises) |
| `search` | 600s | `LogisticRegression` baseline |
| `train` | 600s | Search winner |
| `evaluate` | 60s | Available predictions |
| `report` | 30s | None (optional) |
| `save` | 30s | Best-effort |

---

## Model Artifacts

Each trained model saved to `model_registry/<model_id>/`:

```
model_registry/a1b2c3d4/
├── model.keras / model.pkl    # Trained model
├── preprocessor.pkl           # Fitted preprocessor
├── meta.json                  # Complete metadata (metrics, thresholds, FI, narrative)
├── training_metrics.json      # Accuracy, precision, recall, F1, confusion matrix
├── classification_report.txt  # sklearn classification_report
├── model_selection_explanation.txt
├── training_explanation.txt
├── feature_importance.json    # Permutation importance (tabular)
├── narrative.txt              # Plain-English summary
├── report.html                # Interactive HTML report
├── report.pdf                 # PDF report (if available)
├── plots/                     # loss.png, accuracy.png, confusion_matrix.png
└── circuit_breaker.jsonl      # Stage execution log
```

---

## Testing

```bash
# Full test suite (72 tests, ~17s)
uv run pytest tests/ -v

# Unit tests only (skip integration)
uv run pytest tests/ --ignore=tests/integration -v

# Specific module
uv run pytest tests/test_search.py -v
uv run pytest tests/test_circuit_breaker.py -v
```

**Test coverage:** Target detection, search, multiclass, thresholds, circuit breaker, bugfixes, explainability, HTML reports, quality, validation, and 3 integration suites (tabular, text, image).

---

## Limitations

- **Classification only** - No regression, multi-label, or mixed-modal
- **Local execution** - No hosted cloud; artifacts in `model_registry/`
- **Tabular models** - `GradientBoostingClassifier` (sklearn); LightGBM/XGBoost not included (compile dependency)
- **Text models** - TF-IDF + sklearn; DistilBERT deferred (offline, no HF downloads)
- **Streamlit training** - Synchronous; not suited for long jobs, multi-user, or GPU scheduling
- **Feature count** - Designed for ≤1K features; 10K+ requires external dimensionality reduction

---

## Roadmap

- Config-driven candidate definitions (YAML)
- Async pipeline with true process-level cancellation
- Hugging Face model card export
- Time-series CV split option
- Dockerized prediction API

---

## License

MIT - see [LICENSE](LICENSE)