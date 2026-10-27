"""Small text helpers of the engine."""

__all__ = ['describe_error', 'qualified_name']


def describe_error(*, error: BaseException) -> str:
    """Return ``"ExceptionType: message"``, the form stored in ``status.json``.

    Args:
        error: Any exception.
    """
    return f'{type(error).__name__}: {error}'


def qualified_name(*, cls: type) -> str:
    """Return ``module.QualName`` of a class, the spelling used in files on disk.

    Args:
        cls: Any class.
    """
    return f'{cls.__module__}.{cls.__qualname__}'
