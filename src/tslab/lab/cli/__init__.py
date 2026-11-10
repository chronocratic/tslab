"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from collections.abc import Sequence

from tslab.lab.cli.entrypoint import main as _main

__all__ = ['main']


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the lab alone: ``tslab new`` offers only the ``lab`` template.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
    """
    _main(args)
