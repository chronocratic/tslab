"""The ``ts`` project template of ``tslab new``: time series experiments with chronocratic."""

from importlib import resources

from tslab.lab.cli.new import ProjectTemplate

__all__ = ['TS_TEMPLATE']

TS_TEMPLATE = ProjectTemplate(
    name='ts',
    files=resources.files('tslab.ts') / 'templates' / 'project',
    next_steps=(
        'uv sync',
        'uv run python -m {{package_name}}.toy_data data',
        'then follow README.md',
    ),
)
