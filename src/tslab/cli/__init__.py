"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from collections.abc import Sequence
import dataclasses
import logging
import sys
from typing import Any, cast, Protocol

import tyro

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
    logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
    argv = list(sys.argv[1:]) if args is None else list(args)
    head, extra_args = _split_pass_through(argv=argv)
    command: Command = _parse(argv=head)
    if extra_args and not _takes_pass_through(command=command):
        _LOGGER.error('this command does not take arguments after --')
        raise SystemExit(2)
    if extra_args:
        # ty: ignore[unsound-assignment]
        # — the check above guarantees a dataclass with the extra_args field
        command = dataclasses.replace(cast('Any', command), extra_args=extra_args)
    try:
        code = command.run()
    except UsageError as error:
        failure = error  # logged below, without a traceback: it is a usage error
    else:
        raise SystemExit(code)
    _LOGGER.error('%s', failure)
    raise SystemExit(2)


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
        # ty: ignore[no-matching-overload, unsound-return-statement]
        # — ty sees the dict of dataclass types as no Callable overload match
        return tyro.extras.subcommand_cli_from_dict(
            STUDY_COMMANDS, prog='tslab study', args=list(argv[1:])
        )
    return tyro.extras.subcommand_cli_from_dict(
        {'new': New, 'status': Status, 'study': _Study, 'wandb-sync': WandbSync},
        prog='tslab',
        args=list(argv),
    )
