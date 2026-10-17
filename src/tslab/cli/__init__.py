"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from collections.abc import Sequence
import logging
import sys
from typing import Protocol

import tyro

from tslab.cli.new import New
from tslab.cli.status import Status
from tslab.cli.wandb_sync import WandbSync

__all__ = ['main']


class Command(Protocol):
    """A ``tslab`` subcommand: a dataclass parsed by tyro, with a ``run`` method."""

    def run(self) -> int:
        """Execute the command and return the process exit code."""
        ...


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the ``tslab`` console script.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.

    Raises:
        SystemExit: Always, with the subcommand's exit code.
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
    command: Command = tyro.extras.subcommand_cli_from_dict(
        {'new': New, 'status': Status, 'wandb-sync': WandbSync}, prog='tslab', args=args
    )
    raise SystemExit(command.run())
