"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import inspect
import logging
import sys
from typing import Annotated, Any, cast, Protocol, Union

import tyro

from tslab.cli import output
from tslab.cli.console import configure_logging, use_stderr
from tslab.cli.doctor import Doctor
from tslab.cli.machine import MACHINE_COMMANDS
from tslab.cli.new import New
from tslab.cli.status import Status
from tslab.cli.study import STUDY_COMMANDS
from tslab.cli.usage import UsageError
from tslab.cli.wandb_sync import WandbSync

__all__ = ['main']

_LOGGER = logging.getLogger(__name__)


class Command(Protocol):
    """A ``tslab`` subcommand: a dataclass parsed by tyro, with a ``run`` method."""

    def run(self) -> int:
        """Execute the command and return the process exit code."""
        ...


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
    json_mode = '--json' in head
    output.reset_document_state()
    if json_mode:
        use_stderr()
    try:
        command: Command = _parse(argv=head)
    except SystemExit as error:
        code = _system_exit_code(error=error)
        _print_error_document(
            json_mode=json_mode,
            exit_code=code,
            message=f'command argument parsing failed with exit code {code}',
        )
        raise
    if extra_args and not _takes_pass_through(command=command):
        _LOGGER.error('this command does not take arguments after --')
        _print_error_document(
            json_mode=json_mode,
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
            json_mode=json_mode,
            exit_code=exit_code,
            message=f'command exited with code {exit_code}',
        )
        raise
    except UsageError as error:
        failure = error  # logged below, without a traceback: it is a usage error
    else:
        _print_error_document(
            json_mode=json_mode, exit_code=code, message=f'command exited with code {code}'
        )
        raise SystemExit(code)
    _LOGGER.error('%s', failure)
    _print_error_document(json_mode=json_mode, exit_code=2, message=str(failure))
    raise SystemExit(2)


def _system_exit_code(*, error: SystemExit) -> int:
    if isinstance(error.code, int):
        return error.code
    return 0 if error.code is None else 1


def _print_error_document(*, json_mode: bool, exit_code: int, message: str) -> None:
    if json_mode and exit_code != 0 and not output.has_printed_document():
        output.print_json(document=output.error_document(exit_code=exit_code, message=message))


def _split_pass_through(*, argv: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """The arguments before the first ``--``, and the rest."""
    if '--' in argv:
        position = argv.index('--')
        return list(argv[:position]), tuple(argv[position + 1 :])
    return list(argv), ()


def _takes_pass_through(*, command: object) -> bool:
    return dataclasses.is_dataclass(command) and hasattr(command, 'extra_args')


def _parse(*, argv: Sequence[str]) -> Command:
    if argv and argv[0] == 'study':
        return _cli(
            commands=STUDY_COMMANDS,
            prog='tslab study',
            args=argv[1:],
            description=(
                'Launch and follow the runs of a study. STUDY is module:attribute, for example '
                'my_paper.studies:example; --machine is a name from tslab machine list, '
                'machines/<name>.toml above the working folder, or a profile path.'
            ),
        )
    if argv and argv[0] == 'machine':
        return _cli(
            commands=MACHINE_COMMANDS,
            prog='tslab machine',
            args=argv[1:],
            description='Name machine profiles once, then pass --machine <name> from any folder.',
        )
    return _cli(
        commands={
            'machine': _Machine,
            'doctor': Doctor,
            'new': New,
            'status': Status,
            'study': _Study,
            'wandb-sync': WandbSync,
        },
        prog='tslab',
        args=argv,
        description=(
            'Helpers around TSLab runs. Experiments run through your own scripts; tslab\n'
            'creates projects, launches studies and shows what runs are doing.\n\n'
            'A study, step by step:\n'
            '  tslab new my-paper                                            create a project\n'
            '  tslab study submit STUDY --machine local --dry-run            check the plan\n'
            '  tslab study submit STUDY --machine local                      launch the runs\n'
            '  tslab study status STUDY --machine local                      follow them\n'
            '  tslab study wait STUDY --machine local                        '
            'wait until done or broken\n'
            '  tslab study failures STUDY --machine local                    show errors and logs\n'
            '  tslab study resubmit STUDY --machine local                    '
            'retry interrupted runs\n'
            '  tslab study results STUDY --machine local --out results.csv   collect metrics\n\n'
            'STUDY is module:attribute, for example my_paper.studies:example.'
        ),
    )


def _cli(
    *, commands: Mapping[str, type], prog: str, args: Sequence[str], description: str
) -> Command:
    """Parse one subcommand from ``commands``.

    Without a subcommand in ``args`` tyro prints the list of commands; each
    entry then shows only the first docstring line, so the list stays one line
    per command. The full docstring, with its examples, is the help of the command.
    """
    listing = not any(arg in commands for arg in args)
    options = tuple(
        Annotated[
            Any,
            tyro.conf.subcommand(
                name=name,
                constructor=command,
                description=_summary(command=command) if listing else None,
            ),
        ]
        for name, command in commands.items()
    )
    # why: every option constructs one of the command dataclasses, each with a run method
    return cast(
        'Command',
        tyro.cli(
            cast('Any', Union[options]),  # noqa: UP007 # why: `|` cannot unpack a runtime tuple
            prog=prog,
            args=list(args),
            config=(tyro.conf.FlagCreatePairsOff,),
            description=description,
        ),
    )


def _summary(*, command: type) -> str:
    return (inspect.getdoc(command) or '').split('\n', 1)[0]
