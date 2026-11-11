"""Build the tyro command tree from the subcommand classes.

Every ``tslab`` command is a frozen dataclass with a ``run`` method. This module
assembles them into the nested subcommand groups tyro parses, and owns the
pass-through (``--``) argument handling.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import inspect
import pkgutil
from types import MappingProxyType
from typing import Annotated, Any, cast, Protocol, Union

import tyro

from tslab.lab.cli.doctor import Doctor
from tslab.lab.cli.machine import MACHINE_COMMANDS
from tslab.lab.cli.new import New
from tslab.lab.cli.status import Status
from tslab.lab.cli.study import _INTERNAL_STUDY_COMMANDS, STUDY_COMMANDS

__all__ = ['Command', 'LazyCommand']


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
_COMMANDS: dict[str, type] = {
    _MACHINE: _Machine,
    'doctor': Doctor,
    'new': New,
    'status': Status,
    _STUDY: _Study,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyCommand:
    """A command that is listed from a summary and imported only when it is typed."""

    summary: str
    target: str  # "package.module:ClassName"

    def load(self) -> type:
        """Import the command class."""
        return cast('type', pkgutil.resolve_name(self.target))

    def listing(self) -> type:
        """A stand-in with the summary as its docstring, for the list of commands."""
        stand_in = dataclasses.make_dataclass(
            'ListedCommand', [], frozen=True, kw_only=True, namespace={'run': _never_run}
        )
        stand_in.__doc__ = self.summary
        return stand_in


def _never_run(self: object) -> int:  # noqa: ARG001 # why: a method of the listing stand-in
    # why: never reached; `_parse` swaps in the real command when its name is typed
    msg = 'unreachable'
    raise AssertionError(msg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Group:
    """A command group: visible and hidden commands, its program name and help."""

    commands: Mapping[str, type]
    hidden_commands: Mapping[str, type]
    prog: str
    description: str


_GROUPS: Mapping[str, _Group] = MappingProxyType(
    {
        _STUDY: _Group(
            commands=STUDY_COMMANDS,
            hidden_commands=_INTERNAL_STUDY_COMMANDS,
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
            hidden_commands=MappingProxyType({}),
            prog=f'tslab {_MACHINE}',
            description='Name machine profiles once, then pass --machine <name> from any folder.',
        ),
    }
)


def _parse(*, argv: Sequence[str], extra_commands: Mapping[str, type | LazyCommand]) -> Command:
    group = _GROUPS.get(argv[0]) if argv else None
    if group is not None:
        hidden = group.hidden_commands if len(argv) > 1 and argv[1] in group.hidden_commands else {}
        return _cli(
            commands={**group.commands, **hidden},
            prog=group.prog,
            args=argv[1:],
            description=group.description,
        )
    return _cli(
        commands=_top_level_commands(typed=argv[0] if argv else None, extra=extra_commands),
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


def check_command_clash(*, extra: Mapping[str, type | LazyCommand]) -> None:
    """Raise ``ValueError`` when a layer adds a command whose name a lab command already has."""
    clash = sorted(set(extra) & set(_COMMANDS))
    if clash:
        message = f'commands {clash} are already lab commands'
        raise ValueError(message)


def _top_level_commands(
    *, typed: str | None, extra: Mapping[str, type | LazyCommand]
) -> dict[str, type]:
    """The top-level commands; a lazy one is imported only when it is the one typed.

    The caller has checked the names for a clash with `check_command_clash`.
    """
    added = {
        name: (entry.load() if name == typed else entry.listing())
        if isinstance(entry, LazyCommand)
        else entry
        for name, entry in extra.items()
    }
    return dict(sorted({**_COMMANDS, **added}.items()))


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
