"""JSON output helpers for the command line."""

from collections.abc import Mapping
import json
import sys

__all__ = ['error_document', 'has_printed_document', 'print_json', 'reset_document_state']

_document_state = [False]


def print_json(*, document: Mapping[str, object]) -> None:
    """Print one JSON document to standard output.

    Args:
        document: JSON-compatible document to print.

    Returns:
        None.
    """
    sys.stdout.write(json.dumps(document, indent=2) + '\n')
    _document_state[0] = True


def error_document(*, exit_code: int, message: str) -> dict[str, object]:
    """Build the standard error document.

    Args:
        exit_code: Process exit code for the error.
        message: Human-readable error message.

    Returns:
        The error document.
    """
    # When API is stable: version this id (tslab.error/1); bump on a breaking change.
    return {'schema': 'tslab.error', 'exit_code': exit_code, 'message': message}


def reset_document_state() -> None:
    """Reset whether the current command printed its document.

    Returns:
        None.
    """
    _document_state[0] = False


def has_printed_document() -> bool:
    """Return whether the current command printed its document.

    Returns:
        Whether ``print_json`` has written a document since the last reset.
    """
    return _document_state[0]
