"""Adapter for chronocratic-models (``pip install tslab[chronocratic-models]``).

Every chronocratic model is offered on the command line with its own settings,
generated from its parameter class. Augmentations are given as recipes.

Known limitation: chronocratic-models 0.1.0a20 draws part of the training
randomness of TS2Vec, AutoTCL and CoST (training crops, encoder masks, TS2Vec's
crop-and-shift views) from NumPy generators that no seed controls, so two
trainings with the same seed can differ.
"""

from tslab.adapters.chronocratic_models.settings import (
    chronocratic_model_choice,
    ChronocraticModelChoice,
    DEFAULT_AUGMENTATION,
    entry_of,
    model_settings,
    ModelEntry,
    MODELS,
    recipe_subcommand,
    recipe_view_type,
    RUNTIME_FIELDS,
)

__all__ = [
    'DEFAULT_AUGMENTATION',
    'MODELS',
    'RUNTIME_FIELDS',
    'ChronocraticModelChoice',
    'ModelEntry',
    'chronocratic_model_choice',
    'entry_of',
    'model_settings',
    'recipe_subcommand',
    'recipe_view_type',
]
