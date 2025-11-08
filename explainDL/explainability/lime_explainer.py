"""
lime_explainer.py
-----------------
LIME-based local explanations for tabular or text models.
"""

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def explain_with_lime(model, X_train, X_instance, feature_names, class_names=None, num_features=10):
    """
    Generates LIME explanation for a single prediction and returns both a matplotlib figure
    and a small dataframe with feature contributions.
    """
    X_train_arr = np.array(X_train)
    X_instance_arr = np.array(X_instance)

    explainer = LimeTabularExplainer(
        training_data=X_train_arr,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    exp = explainer.explain_instance(X_instance_arr, model.predict, num_features=num_features)
    fig = exp.as_pyplot_figure()
    plt.tight_layout()

    # Also return a dataframe for numeric display
    local_exp = exp.as_list()
    lime_df = pd.DataFrame(local_exp, columns=['Feature', 'Contribution'])
    return fig, lime_df
