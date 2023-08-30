"""Define custom exceptions used in the project."""


class MissingEnvironmentVariableError(Exception):
    """Raised when an environment variable is missing."""

    pass


class MissingDataError(Exception):
    """Raised when expected data is missing."""

    pass
