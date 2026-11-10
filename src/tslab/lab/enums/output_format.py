"""The ways a command writes its output."""

from enum import StrEnum

__all__ = ['OutputFormat']


class OutputFormat(StrEnum):
    """How a command prints to the terminal.

    Attributes:
        TEXT: Plain text for people.
        JSON: A JSON document, selected by the ``--json`` flag.
    """

    TEXT = 'text'
    JSON = 'json'
