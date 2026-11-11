"""The ``tslab`` command-line entry point."""

from collections.abc import Mapping, Sequence
import dataclasses
import logging
import sys
from typing import Any, cast

from tslab.lab.cli.common import output
from tslab.lab.cli.common.console import configure_logging, use_stderr
from tslab.lab.cli.common.errors import UsageError
from tslab.lab.cli.new import New, ProjectTemplate
from tslab.lab.cli.parser import (
    _parse,
    _split_pass_through,
    _system_exit_code,
    _takes_pass_through,
    check_command_clash,
    Command,
    LazyCommand,
)
from tslab.lab.enums import OutputFormat

__all__ = ['main']

_LOGGER = logging.getLogger('tslab.lab.cli')


def main(
    args: Sequence[str] | None = None,
    *,
    extra_commands: Mapping[str, type | LazyCommand] | None = None,
    templates: Sequence[ProjectTemplate] | None = None,
    default_template: str | None = None,
) -> None:
    """Entry point of the ``tslab`` console script.

    Arguments after the first ``--`` are pass-through: tslab does not parse
    them and hands them to the ``extra_args`` field of the command. A ``--`` on a
    command without that field exits 2.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
        extra_commands: Top-level commands a layer above the lab adds, by name.
        templates: Project templates for ``tslab new``; ``None`` means the lab template alone.
        default_template: Template used without ``--template``; ``None`` means the lab template.

    Raises:
        ValueError: If an extra command has the name of a lab command.
        SystemExit: Always, with the exit code of the subcommand.
    """
    # why: a programmer error, raised before parsing so --json mode never meets it mid-parse
    check_command_clash(extra=extra_commands or {})
    configure_logging()
    argv = list(sys.argv[1:]) if args is None else list(args)
    head, extra_args = _split_pass_through(argv=argv)
    output_format = OutputFormat.JSON if '--json' in head else OutputFormat.TEXT
    output.reset_document_state()
    if output_format is OutputFormat.JSON:
        use_stderr()
    try:
        command: Command = _parse(argv=head, extra_commands=extra_commands or {})
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
        # why: the check above guarantees a dataclass with the extra_args field
        command = cast('Command', dataclasses.replace(cast('Any', command), extra_args=extra_args))
    try:
        code = _run(command=command, templates=templates, default_template=default_template)
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


def _run(
    *, command: Command, templates: Sequence[ProjectTemplate] | None, default_template: str | None
) -> int:
    if isinstance(command, New) and templates is not None:
        return command.run(
            templates=tuple(templates), default_template=default_template or templates[0].name
        )
    return command.run()


def _print_error_document(*, output_format: OutputFormat, exit_code: int, message: str) -> None:
    if output_format is OutputFormat.JSON and exit_code != 0 and not output.has_printed_document():
        output.print_json(document=output.error_document(exit_code=exit_code, message=message))
