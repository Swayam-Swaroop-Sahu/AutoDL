# src/explainability/lime_explainer.py
"""
LIME explainability for tabular and text models.

Provides:
- explain_tabular_with_lime(model, X_train, instance, feature_names, class_names=None, num_features=10)
- explain_text_with_lime(model, tokenizer, texts, class_names=None, num_features=10, index_to_explain=0, max_len=100)

Notes:
- lime package must be installed to use these functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTabularExplainer = None
    LimeTextExplainer = None


def explain_tabular_with_lime(
    model,
    X_train,
    instance,
    feature_names,
    class_names=None,
    num_features=10,
):
    """
    LIME explanation for a single tabular instance.

    Returns:
        fig : matplotlib.figure.Figure
        contrib_df : pandas.DataFrame
    """
    if LimeTabularExplainer is None:
        raise ImportError("lime is not installed. Install lime to use LIME explainability.")

    X_train_arr = np.array(X_train)
    instance_arr = np.array(instance)

    explainer = LimeTabularExplainer(
        training_data=X_train_arr,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    exp = explainer.explain_instance(
        instance_arr,
        model.predict,
        num_features=num_features
    )

    fig = exp.as_pyplot_figure()
    plt.tight_layout()

    contrib_list = exp.as_list()
    contrib_df = pd.DataFrame(contrib_list, columns=["Feature", "Contribution"])
    return fig, contrib_df


def explain_text_with_lime(
    model,
    tokenizer,
    texts,
    class_names=None,
    num_features=10,
    index_to_explain=0,
    max_len=100,
):
    """
    LIME explanation for a single text instance.

    tokenizer: keras.preprocessing.text.Tokenizer (used internally for prediction)
    texts: list[str] (original raw texts)
    """

    if LimeTextExplainer is None:
        raise ImportError("lime is not installed. Install lime to use LIME explainability.")

    def predict_proba(raw_texts):
        # Convert raw_texts to padded sequences using tokenizer; return model.predict
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seqs = tokenizer.texts_to_sequences(raw_texts)
        padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
        preds = model.predict(padded)
        return preds

    explainer = LimeTextExplainer(class_names=class_names)

    target_text = texts[index_to_explain]
    exp = explainer.explain_instance(
        target_text,
        predict_proba,
        num_features=num_features
    )

    fig = exp.as_pyplot_figure()
    plt.tight_layout()

    contrib_df = pd.DataFrame(exp.as_list(), columns=["Token/Pattern", "Contribution"])
    return fig, contrib_df
