# src/explainability/text_explainer.py
"""
High-level text explainability helpers.

Provides convenience wrapper `explain_text_sample(...)` that calls LIME wrapper
and returns a consistent (fig, contrib_df) pair.
"""

import pandas as pd
from .lime_explainer import explain_text_with_lime


def explain_text_sample(
    model,
    tokenizer,
    texts,
    class_names=None,
    num_features=10,
    index_to_explain=0,
    max_len=100,
):
    """
    Convenience wrapper around LIME text explanation.

    Returns:
        fig : matplotlib.figure.Figure
        contrib_df : pandas.DataFrame
    """

    fig, contrib_df = explain_text_with_lime(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        class_names=class_names,
        num_features=num_features,
        index_to_explain=index_to_explain,
        max_len=max_len,
    )

    if not isinstance(contrib_df, pd.DataFrame):
        contrib_df = pd.DataFrame(contrib_df, columns=["Token/Pattern", "Contribution"])

    return fig, contrib_df
