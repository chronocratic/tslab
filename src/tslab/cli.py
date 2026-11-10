"""The ``tslab`` command-line tool, with the commands and templates of every layer."""

from collections.abc import Sequence

from tslab.lab.cli.entrypoint import main as lab_main
from tslab.lab.cli.new import LAB_TEMPLATE, register_template
from tslab.lab.cli.parser import register_command
from tslab.ml.cli.wandb_sync import WandbSync
from tslab.ts.project_template import TS_TEMPLATE

__all__ = ['main']


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the ``tslab`` console script.

    Adds the ``wandb-sync`` command and the ``lab`` and ``ts`` project templates
    (``ts`` is the default) to the lab command line, then runs it.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
    """
    register_command(name='wandb-sync', command=WandbSync)
    register_template(template=LAB_TEMPLATE)
    register_template(template=TS_TEMPLATE, default=True)
    lab_main(args)
