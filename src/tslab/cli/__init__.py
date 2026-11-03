"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from collections.abc import Sequence
import dataclasses
import logging
import sys
from typing import Any, cast

from tslab.cli.common import configure_logging, output, UsageError, use_stderr
from tslab.enums import OutputFormat

__all__ = ['main']

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Study:
    """Run a study of experiments: submit, status, resubmit, results."""

    def run(self) -> int:
        # why: never reached; `_parse` hands every `study` call to the study group
        msg = 'unreachable'
        raise AssertionError(msg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Machine:
    """Register machine profiles under names: add, list, remove."""

    def run(self) -> int:
        # why: never reached; `_parse` hands every `machine` call to the machine group
        msg = 'unreachable'
        raise AssertionError(msg)


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the ``tslab`` console script.

    Arguments after the first ``--`` are pass-through: tslab does not parse
    them and hands them to the ``extra_args`` field of the command. A ``--`` on a
    command without that field exits 2.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.

    Raises:
        SystemExit: Always, with the exit code of the subcommand.
    """
    configure_logging()
    argv = list(sys.argv[1:]) if args is None else list(args)
    head, extra_args = _split_pass_through(argv=argv)
    output_format = OutputFormat.JSON if '--json' in head else OutputFormat.TEXT
    output.reset_document_state()
    if output_format is OutputFormat.JSON:
        use_stderr()
    try:
        command: Command = _parse(argv=head)
    except SystemExit as error:
        code = _system_exit_code(error=error)
        _print_error_document(
            output_format=output_format,
            exit_code=code,
            message=f'command argument parsing failed with exit code {code}',
        )
        raise
    if extra_args and not _takes_pass_through(command=command):
        _LOGGER.error('this command does not take arguments after --')
        _print_error_document(
            output_format=output_format,
            exit_code=2,
            message='this command does not take arguments after --',
        )
        raise SystemExit(2)
    if extra_args:
        # ty: ignore[unsound-assignment]
        # — the check above guarantees a dataclass with the extra_args field
        command = dataclasses.replace(cast('Any', command), extra_args=extra_args)
    try:
        code = command.run()
    except SystemExit as error:
        exit_code = _system_exit_code(error=error)
        _print_error_document(
            output_format=output_format,
            exit_code=exit_code,
            message=f'command exited with code {exit_code}',
        )
        raise
    except UsageError as error:
        failure = error  # logged below, without a traceback: it is a usage error
    except Exception as error:  # why: JSON commands need a document for every failure
        _print_error_document(output_format=output_format, exit_code=1, message=str(error))
        raise
    else:
        _print_error_document(
            output_format=output_format, exit_code=code, message=f'command exited with code {code}'
        )
        raise SystemExit(code)
    _LOGGER.error('%s', failure)
    _print_error_document(output_format=output_format, exit_code=2, message=str(failure))
    raise SystemExit(2)


def _print_error_document(*, output_format: OutputFormat, exit_code: int, message: str) -> None:
    if output_format is OutputFormat.JSON and exit_code != 0 and not output.has_printed_document():
        output.print_json(document=output.error_document(exit_code=exit_code, message=message))


# why: the parser imports `_Study` and `_Machine` from this door, so the import
# comes after the class definitions
from tslab.cli.parser import (  # noqa: E402
    _parse,
    _split_pass_through,
    _system_exit_code,
    _takes_pass_through,
    Command,
)
