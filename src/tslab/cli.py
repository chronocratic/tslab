"""The ``tslab`` command-line tool, with the commands and templates of every layer."""

from collections.abc import Sequence
from importlib import resources

from tslab.lab.cli.entrypoint import main as lab_main
from tslab.lab.cli.new import LAB_TEMPLATE, ProjectTemplate
from tslab.lab.cli.parser import LazyCommand

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


def main(args: Sequence[str] | None = None) -> None:
    """Entry point of the ``tslab`` console script.

    Adds the ``wandb-sync`` command and the ``ts`` project template (the default) to the
    lab command line, then runs it.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
    """
    lab_main(
        args,
        extra_commands={
            'wandb-sync': LazyCommand(
                summary='Resync local W&B runs that the server shows as crashed or failed.',
                target='tslab.ml.cli.wandb_sync:WandbSync',
            )
        },
        templates=(LAB_TEMPLATE, TS_TEMPLATE),
        default_template=TS_TEMPLATE.name,
    )
