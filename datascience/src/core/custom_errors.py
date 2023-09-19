"""Define custom exceptions used in the project."""


class MissingEnvironmentVariableError(Exception):
    """Raised when an environment variable is missing."""

    pass


class MissingDataError(Exception):
    """Raised when expected data is missing."""

    pass


class ImageProcessingError(Exception):
    """Raised when an issue occurs during image processing."""

    pass


class RequirementsGenerationError(Exception):
    """Raised when an error occurs during Poetry's requirements.txt generation."""

    pass
