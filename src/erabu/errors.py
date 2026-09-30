"""Errors raised before a solver can be called."""


class ModelSizeError(ValueError):
    """The conservative pair expansion estimate exceeds the requested limit."""


class NumericalError(ValueError):
    """A model calculation produced a non-finite value."""
