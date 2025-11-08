"""
visualization.py
----------------
General visualization utilities for model results, metrics,
and comparison plots.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_history(history):
    """
    Plots model training accuracy and loss curves.
    """
    if not history:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy plot
    ax[0].plot(history.get('accuracy', []), label='Train Acc')
    ax[0].plot(history.get('val_accuracy', []), label='Val Acc')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss plot
    ax[1].plot(history.get('loss', []), label='Train Loss')
    ax[1].plot(history.get('val_loss', []), label='Val Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()
    return fig


def plot_model_comparison(results_list):
    """
    Compares multiple models' accuracies visually.

    Parameters
    ----------
    results_list : list[dict]
        Each dict must contain 'model_name' and 'accuracy' keys.
    """
    df = pd.DataFrame(results_list)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["model_name"], df["accuracy"], color='steelblue')
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Comparison")
    plt.tight_layout()
    return fig
