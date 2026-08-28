"""Typed failures surfaced by pipeline stages."""


class ShortEngineError(RuntimeError):
    """Base class for expected engine failures."""


class InputError(ShortEngineError):
    """The requested source or option is invalid."""


class DependencyError(ShortEngineError):
    """A required local dependency is unavailable."""


class MediaError(ShortEngineError):
    """Media probing or conversion failed."""


class InferenceError(ShortEngineError):
    """A model could not complete inference."""


class ModelOutputError(InferenceError):
    """A model response violated its schema."""


class RenderError(ShortEngineError):
    """A clip could not be rendered or validated."""
