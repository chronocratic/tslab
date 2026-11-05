"""Grid builders shared by the template studies."""

from tslab.study import Grid, Override, setting_axis

from {{package_name}} import classify, forecast

from .datasets import TOY_ETT, TOY_UCR
from .recipes import classification_recipes, forecasting_recipes

__all__ = ['classification_grid', 'forecasting_grid']


def classification_grid(*, seeds: tuple[int, ...]) -> Grid:
    """Build the shared classification grid for the requested seeds.

    Args:
        seeds: The seeds to run.

    Returns:
        The classification grid over the toy UCR dataset and recipe.
    """
    return Grid(
        short_name='cls',
        command=('uv', 'run', 'python', '-m', '{{package_name}}.classify', 'no-preset'),
        config=classify.ClassifyConfig,
        axes={
            'recipe': classification_recipes(),
            'dataset': TOY_UCR,
            'seed': setting_axis(path='seed', values=seeds),
        },
        run_path='{grid}/{recipe}/{dataset}/seed_{seed}',
    )


def forecasting_grid(*, seeds: tuple[int, ...]) -> Grid:
    """Build the shared forecasting grid for the requested seeds.

    Args:
        seeds: The seeds to run.

    Returns:
        The forecasting grid over the toy ETT variant and recipes.
    """
    return Grid(
        short_name='fc',
        command=('uv', 'run', 'python', '-m', '{{package_name}}.forecast', 'no-preset'),
        config=forecast.ForecastConfig,
        axes={
            'recipe': forecasting_recipes(),
            'variant': TOY_ETT,
            'seed': setting_axis(path='seed', values=seeds),
        },
        run_path='{grid}/{recipe}/{variant}/seed_{seed}',
        overrides=(Override(match={'recipe': 'cost-cropped'}, settings={'data.batch_size': 1}),),
    )
