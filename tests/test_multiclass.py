"""Tests for unified multiclass code path (Phase 1c).

Verifies binary, 3-class, and 5-class tabular datasets all train and predict
through the same code path: softmax head + sparse_categorical_crossentropy +
argmax decode. No num_classes==2 branches remain.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.model_selection.tabular_models import build_mlp_small


def _make_csv_df(n_samples=200, n_classes=2, seed=42):
    """Generate a synthetic DataFrame with a known number of classes."""
    n_features = 6
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features, n_classes),
        n_redundant=0,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["target"] = y
    return df


# ---------------------------------------------------------------------------
# Binary (2-class) test
# ---------------------------------------------------------------------------
def test_binary_trains_and_predicts():
    """Binary dataset trains and predicts correctly through the unified path."""
    df = _make_csv_df(n_samples=200, n_classes=2)
    preprocessor = TabularPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col="target")
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert X.shape[1] == 6

    model = build_mlp_small(X.shape[1], num_classes=2)
    # Verify unified softmax head
    assert model.output_shape == (None, 2)
    assert model.loss == "sparse_categorical_crossentropy"

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy() if hasattr(X, "to_numpy") else X, y,
        test_size=0.2, random_state=42, stratify=y,
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=16, verbose=0)

    proba = model.predict(X_val, verbose=0)
    assert proba.shape == (len(X_val), 2), f"Expected (N, 2), got {proba.shape}"
    preds = np.argmax(proba, axis=1)
    assert preds.shape == (len(X_val),)
    assert set(preds.tolist()) <= {0, 1}
    # Class names from LabelEncoder should round-trip
    decoded = preprocessor.target_encoder.inverse_transform(preds)
    assert len(decoded) == len(preds)


# ---------------------------------------------------------------------------
# 3-class test
# ---------------------------------------------------------------------------
def test_three_class_trains_and_predicts():
    """3-class dataset trains and predicts correctly through the unified path."""
    df = _make_csv_df(n_samples=300, n_classes=3)
    preprocessor = TabularPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col="target")
    assert set(np.unique(y).tolist()) <= {0, 1, 2}

    model = build_mlp_small(X.shape[1], num_classes=3)
    assert model.output_shape == (None, 3)
    assert model.loss == "sparse_categorical_crossentropy"

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy() if hasattr(X, "to_numpy") else X, y,
        test_size=0.2, random_state=42, stratify=y,
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=16, verbose=0)

    proba = model.predict(X_val, verbose=0)
    assert proba.shape == (len(X_val), 3), f"Expected (N, 3), got {proba.shape}"
    preds = np.argmax(proba, axis=1)
    assert preds.shape == (len(X_val),)
    assert set(preds.tolist()) <= {0, 1, 2}
    decoded = preprocessor.target_encoder.inverse_transform(preds)
    assert len(decoded) == len(preds)


# ---------------------------------------------------------------------------
# 5-class test
# ---------------------------------------------------------------------------
def test_five_class_trains_and_predicts():
    """5-class dataset trains and predicts correctly through the unified path."""
    df = _make_csv_df(n_samples=500, n_classes=5)
    preprocessor = TabularPreprocessor()
    X, y = preprocessor.fit_transform(df, target_col="target")
    assert set(np.unique(y).tolist()) <= {0, 1, 2, 3, 4}

    model = build_mlp_small(X.shape[1], num_classes=5)
    assert model.output_shape == (None, 5)
    assert model.loss == "sparse_categorical_crossentropy"

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy() if hasattr(X, "to_numpy") else X, y,
        test_size=0.2, random_state=42, stratify=y,
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=16, verbose=0)

    proba = model.predict(X_val, verbose=0)
    assert proba.shape == (len(X_val), 5), f"Expected (N, 5), got {proba.shape}"
    preds = np.argmax(proba, axis=1)
    assert preds.shape == (len(X_val),)
    assert set(preds.tolist()) <= {0, 1, 2, 3, 4}
    decoded = preprocessor.target_encoder.inverse_transform(preds)
    assert len(decoded) == len(preds)


# ---------------------------------------------------------------------------
# Image-model unified path test
# ---------------------------------------------------------------------------
def test_image_builders_use_unified_softmax():
    """Image model builders produce a softmax head regardless of num_classes."""
    from src.model_selection.image_models import build_small_cnn, build_mobilenet, build_efficientnet
    # Test for n_classes=2 (formerly had a binary branch)
    m2 = build_small_cnn((32, 32, 3), num_classes=2)
    assert m2.output_shape == (None, 2)
    assert m2.loss == "sparse_categorical_crossentropy"
    # Test for n_classes=5
    m5 = build_small_cnn((32, 32, 3), num_classes=5)
    assert m5.output_shape == (None, 5)
    assert m5.loss == "sparse_categorical_crossentropy"


# ---------------------------------------------------------------------------
# Tabular builder unified path test
# ---------------------------------------------------------------------------
def test_tabular_builders_use_unified_softmax():
    """Tabular model builders use unified softmax for both 2 and 5 classes."""
    from src.model_selection.tabular_models import build_mlp_small, build_mlp_medium, build_mlp_large
    for builder in [build_mlp_small, build_mlp_medium, build_mlp_large]:
        m2 = builder(10, num_classes=2)
        assert m2.output_shape == (None, 2)
        assert m2.loss == "sparse_categorical_crossentropy"
        m5 = builder(10, num_classes=5)
        assert m5.output_shape == (None, 5)
        assert m5.loss == "sparse_categorical_crossentropy"


# ---------------------------------------------------------------------------
# Argmax-decode sanity check (used in trainer + pipeline_predict)
# ---------------------------------------------------------------------------
def test_argmax_decode_roundtrip():
    """proba → argmax → LabelEncoder.inverse_transform is a closed loop."""
    df = _make_csv_df(n_samples=150, n_classes=4)
    preprocessor = TabularPreprocessor()
    _, y = preprocessor.fit_transform(df, target_col="target")
    # Simulate proba output (shape N, 4)
    np.random.seed(0)
    fake_proba = np.random.dirichlet(np.ones(4), size=len(y))
    preds = np.argmax(fake_proba, axis=1)
    decoded = preprocessor.target_encoder.inverse_transform(preds)
    assert len(decoded) == len(y)
    assert set(decoded).issubset(set(preprocessor.target_encoder.classes_))
