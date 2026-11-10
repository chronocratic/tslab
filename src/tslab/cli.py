"""The ``tslab`` command-line tool, with the commands and templates of every layer."""

from collections.abc import Sequence
import dataclasses
from importlib import resources
import sys

from tslab.lab.cli.entrypoint import main as lab_main
from tslab.lab.cli.new import LAB_TEMPLATE, ProjectTemplate, register_template
from tslab.lab.cli.parser import register_command

__all__ = ['TS_TEMPLATE', 'main']

# why: found through the top package, because importing tslab.ts would load torch
TS_TEMPLATE = ProjectTemplate(
    name='ts',
    files=resources.files('tslab') / 'ts' / 'templates' / 'project',
    next_steps=(
        'uv sync',
        'uv run python -m {{package_name}}.toy_data data',
        'then follow README.md',
    ),
)

_WANDB_SYNC = 'wandb-sync'


@dataclasses.dataclass(frozen=True, kw_only=True)
class _WandbSyncListing:
    """Resync local W&B runs that the server shows as crashed or failed."""

    def run(self) -> int:
        # why: never reached; `main` registers the real command when it is typed
        msg = 'unreachable'
        raise AssertionError(msg)


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the ``tslab`` console script.

    Adds the ``wandb-sync`` command and the ``lab`` and ``ts`` project templates
    (``ts`` is the default) to the lab command line, then runs it.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
    """
    argv = sys.argv[1:] if args is None else list(args)
    if argv[:1] == [_WANDB_SYNC]:
        # why: tslab.ml loads torch; only the command that needs it pays for it
        from tslab.ml.cli.wandb_sync import WandbSync  # noqa: PLC0415

        register_command(name=_WANDB_SYNC, command=WandbSync, replace=True)
    else:
        register_command(name=_WANDB_SYNC, command=_WandbSyncListing, replace=True)
    register_template(template=LAB_TEMPLATE)
    register_template(template=TS_TEMPLATE, default=True)
    lab_main(args)
