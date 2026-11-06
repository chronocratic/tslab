"""The probe grid of the chain study."""

from tslab.study import After, Grid, setting_axis

from {{package_name}} import probe
from {{package_name}}.studies.common import classification_recipes, TOY_UCR

__all__ = ['probe_grid']


def probe_grid(*, seeds: tuple[int, ...]) -> Grid:
    """Build the probe grid that runs after the pretrain grid of the study.

    Each probe run waits for the pretrain run with the same recipe, dataset
    and seed, and loads the checkpoint that run saved.

    Args:
        seeds: The seeds to run.

    Returns:
        The probe grid over the toy UCR dataset, recipe and probe size.
    """
    return Grid(
        short_name='probe',
        command=('uv', 'run', 'python', '-m', '{{package_name}}.probe'),
        config=probe.ProbeConfig,
        axes={
            'recipe': classification_recipes(),
            'dataset': TOY_UCR,
            'seed': setting_axis(path='seed', values=seeds),
            'head': {'small': {'max_train_samples': 100}, 'full': {'max_train_samples': 10_000}},
        },
        after=(After(grid='pretrain'),),
        run_settings={'checkpoint': '{after}/train/default'},
        run_path='{grid}/{recipe}/{dataset}/seed_{seed}/{head}',
    )
