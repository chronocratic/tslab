"""Build the tyro command tree from the subcommand classes.

Every ``tslab`` command is a frozen dataclass with a ``run`` method. This module
assembles them into the nested subcommand groups tyro parses, and owns the
pass-through (``--``) argument handling.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import inspect
from types import MappingProxyType
from typing import Annotated, Any, cast, Protocol, Union

import tyro

from tslab.cli.doctor import Doctor
from tslab.cli.machine import MACHINE_COMMANDS
from tslab.cli.new import New
from tslab.cli.status import Status
from tslab.cli.study import STUDY_COMMANDS
from tslab.cli.wandb_sync import WandbSync

__all__ = ['Command']


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


class Command(Protocol):
    """A ``tslab`` subcommand: a dataclass parsed by tyro, with a ``run`` method."""

    def run(self) -> int:
        """Execute the command and return the process exit code."""
        ...


def _system_exit_code(*, error: SystemExit) -> int:
    if isinstance(error.code, int):
        return error.code
    return 0 if error.code is None else 1


def _split_pass_through(*, argv: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """The arguments before the first ``--``, and the rest."""
    if '--' in argv:
        position = argv.index('--')
        return list(argv[:position]), tuple(argv[position + 1 :])
    return list(argv), ()


def _takes_pass_through(*, command: object) -> bool:
    return dataclasses.is_dataclass(command) and hasattr(command, 'extra_args')


_STUDY = 'study'
_MACHINE = 'machine'


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Group:
    """A command group: the commands under one word, its program name and its help."""

    commands: Mapping[str, type]
    prog: str
    description: str


_GROUPS: Mapping[str, _Group] = MappingProxyType(
    {
        _STUDY: _Group(
            commands=STUDY_COMMANDS,
            prog=f'tslab {_STUDY}',
            description=(
                'Launch and follow the runs of a study. STUDY is a short name, such as example, or '
                'a full module:attribute reference, such as '
                'my_paper.studies.example:study; a short name expands against the studies package '
                'of the nearest tslab.toml; --machine is a name from tslab machine list, '
                'machines/<name>.toml above the working folder, or a profile path.'
            ),
        ),
        _MACHINE: _Group(
            commands=MACHINE_COMMANDS,
            prog=f'tslab {_MACHINE}',
            description='Name machine profiles once, then pass --machine <name> from any folder.',
        ),
    }
)


def _parse(*, argv: Sequence[str]) -> Command:
    group = _GROUPS.get(argv[0]) if argv else None
    if group is not None:
        return _cli(
            commands=group.commands, prog=group.prog, args=argv[1:], description=group.description
        )
    return _cli(
        commands={
            _MACHINE: _Machine,
            'doctor': Doctor,
            'new': New,
            'status': Status,
            _STUDY: _Study,
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
            '  tslab study logs STUDY --machine local --only AXIS=VALUE -f   follow one run log\n'
            '  tslab study resubmit STUDY --machine local                    '
            'retry interrupted runs\n'
            '  tslab study results STUDY --machine local --out results.csv   collect metrics\n\n'
            'STUDY is a short name, such as example, or a full module:attribute reference, such as '
            'my_paper.studies.example:study; a short name expands against the studies package of '
            'tslab.toml.'
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
            config=(tyro.conf.FlagCreatePairsOff, tyro.conf.EnumChoicesFromValues),
            description=description,
        ),
    )


def _summary(*, command: type) -> str:
    return (inspect.getdoc(command) or '').split('\n', 1)[0]
