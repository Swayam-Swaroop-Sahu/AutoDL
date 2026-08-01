"""AutoDL-specific exception hierarchy.

Every error message answers three questions:
  1. What went wrong?
  2. Why did it happen?
  3. What should I do to fix it?
"""


class AutoDLInputError(ValueError):
    """Raised when the user's input data is invalid.

    The user can fix this by adjusting their file/data before re-uploading.
    """


class AutoDLTrainingError(RuntimeError):
    """Raised when the pipeline encounters an internal failure during training.

    This indicates a bug or resource issue that the user cannot fix by
    adjusting their data alone.
    """


class AutoDLTargetAmbiguousError(AutoDLInputError):
    """Raised when target column detection is ambiguous.

    The UI should catch this to present a disambiguation radio button,
    not show a generic error.
    """


class AutoDLConfigError(ValueError):
    """Raised when the configuration (tuning, model selection) is invalid."""