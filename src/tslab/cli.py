"""The ``tslab`` command-line tool, with the commands and templates of every layer."""

from collections.abc import Sequence
from importlib import resources

from tslab.lab.cli.entrypoint import main as lab_main
from tslab.lab.cli.new import ProjectTemplate, register_template, template_scope
from tslab.lab.cli.parser import command_scope, register_lazy_command

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
    lab command line, then runs it. The additions last only for this call, so the
    lab entry point of the same process stays lab-only.

    Args:
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.
    """
    with command_scope(), template_scope():
        register_lazy_command(
            name='wandb-sync',
            summary='Resync local W&B runs that the server shows as crashed or failed.',
            target='tslab.ml.cli.wandb_sync:WandbSync',
        )
        register_template(template=TS_TEMPLATE, default=True)
        lab_main(args)
