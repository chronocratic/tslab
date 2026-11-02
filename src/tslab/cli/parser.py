"""Build the tyro command tree from the subcommand classes.

Every ``tslab`` command is a frozen dataclass with a ``run`` method. This module
assembles them into the nested subcommand groups tyro parses, and owns the
pass-through (``--``) argument handling.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import inspect
from typing import Annotated, Any, cast, Protocol, Union

import tyro

from tslab.cli import _Machine, _Study
from tslab.cli.doctor import Doctor
from tslab.cli.machine import MACHINE_COMMANDS
from tslab.cli.new import New
from tslab.cli.status import Status
from tslab.cli.study import STUDY_COMMANDS
from tslab.cli.wandb_sync import WandbSync

__all__ = ['Command']


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


def _parse(*, argv: Sequence[str]) -> Command:
    if argv and argv[0] == 'study':
        return _cli(
            commands=STUDY_COMMANDS,
            prog='tslab study',
            args=argv[1:],
            description=(
                'Launch and follow the runs of a study. STUDY is module:attribute, for example '
                'my_paper.studies:example; a short name expands against the studies module of '
                'the nearest tslab.toml; --machine is a name from tslab machine list, '
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
            '  tslab study logs STUDY --machine local --only AXIS=VALUE -f   follow one run log\n'
            '  tslab study resubmit STUDY --machine local                    '
            'retry interrupted runs\n'
            '  tslab study results STUDY --machine local --out results.csv   collect metrics\n\n'
            'STUDY is module:attribute, for example my_paper.studies:example; a short name '
            'expands against the studies module of tslab.toml.'
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
