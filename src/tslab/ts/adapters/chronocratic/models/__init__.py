"""Adapter for chronocratic-models (``pip install tslab[chronocratic-models]``).

Every chronocratic model is offered on the command line with its own settings,
generated from its parameter class. Augmentations are given as recipes.
"""

from tslab.ts.adapters.chronocratic.models.build import build_chronocratic_model, FingerprintedModel
from tslab.ts.adapters.chronocratic.models.encoder import ChronocraticEncoder
from tslab.ts.adapters.chronocratic.models.settings import (
    chronocratic_model_choice,
    ChronocraticModelChoice,
    model_settings,
    MODELS,
    recipe_subcommand,
    recipe_view_type,
)

__all__ = [
    'MODELS',
    'ChronocraticEncoder',
    'ChronocraticModelChoice',
    'FingerprintedModel',
    'build_chronocratic_model',
    'chronocratic_model_choice',
    'model_settings',
    'recipe_subcommand',
    'recipe_view_type',
]
