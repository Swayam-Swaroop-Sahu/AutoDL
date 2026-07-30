# src/model_selection/__init__.py

# Lazy-import to avoid circular dependency chain:
#   model_selection.__init__ -> search -> core.config -> core.__init__ -> pipeline_train
# All public names are available via module-qualified imports or the inline functions below.

def _get_search():
    """Lazy accessor for search module."""
    from .search import successive_halving_search, Candidate, DEFAULT_SCORING
    return successive_halving_search, Candidate, DEFAULT_SCORING


def _get_tabular():
    from .tabular_candidates import get_tabular_candidates
    return get_tabular_candidates


def _get_text():
    from .text_candidates import get_text_candidates
    return get_text_candidates


def _get_image():
    from .image_candidates import get_image_candidates
    return get_image_candidates


# Legacy Keras model builders (still used for image transfer learning)
from .tabular_models import build_mlp_small, build_mlp_medium, build_mlp_large
from .image_models import build_small_cnn, build_mobilenet, build_efficientnet
from .text_models import build_lstm, build_bilstm, build_text_cnn

# Legacy selector (kept for backward-compat but deprecated)
from .selector import select_best_model

__all__ = [
    "build_mlp_small",
    "build_mlp_medium",
    "build_mlp_large",
    "build_small_cnn",
    "build_mobilenet",
    "build_efficientnet",
    "build_lstm",
    "build_bilstm",
    "build_text_cnn",
    "select_best_model",
]