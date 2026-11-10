"""Shared command-line helpers: console output, JSON documents and usage errors.

The ``tslab`` commands are thin; this holds what they all print and how they
report a bad command line. Import from the door, not the submodules.
"""

from tslab.lab.cli.common.console import (
    configure_logging,
    CONSOLE,
    ConsoleHandler,
    make_console,
    use_stderr,
    watch,
)
from tslab.lab.cli.common.errors import UsageError
from tslab.lab.cli.common.output import (
    error_document,
    has_printed_document,
    print_json,
    reset_document_state,
)
from tslab.lab.cli.common.project import Request, resolve_request

__all__ = [
    'CONSOLE',
    'ConsoleHandler',
    'Request',
    'UsageError',
    'configure_logging',
    'error_document',
    'has_printed_document',
    'make_console',
    'print_json',
    'reset_document_state',
    'resolve_request',
    'use_stderr',
    'watch',
]
