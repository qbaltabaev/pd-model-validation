"""Package-specific exceptions with actionable validation context."""


class InputValidationError(ValueError):
    """Raised when input data cannot safely be used for model diagnostics."""
