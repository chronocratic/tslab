"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the user's own ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.
"""

from tslab.study.definition import (
    DEFAULT_RUN_SETTINGS,
    Grid,
    match_problems,
    Override,
    run_matches,
    setting_axis,
    Study,
)
from tslab.study.settings import check_paths, from_presets, Settings, settings_to_args

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'Grid',
    'Override',
    'Settings',
    'Study',
    'check_paths',
    'from_presets',
    'match_problems',
    'run_matches',
    'setting_axis',
    'settings_to_args',
]
