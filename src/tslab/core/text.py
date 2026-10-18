"""Small text helpers shared by the engine and the layers above it."""

__all__ = ['describe_error', 'first_line', 'qualified_name']


def first_line(*, text: str) -> str:
    """Return the first line of ``text``, or ``''`` if it has none.

    Args:
        text: Any text, for example an exception message.
    """
    lines = text.splitlines()
    return lines[0] if lines else ''


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
