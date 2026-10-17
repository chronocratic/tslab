"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the user's own ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.
"""

from tslab.study.definition import (
    DEFAULT_RUN_SETTINGS,
    expand_runs,
    Grid,
    match_problems,
    merge_overrides,
    Override,
    run_matches,
    select_runs,
    setting_axis,
    Study,
    StudyRun,
)
from tslab.study.expansion import expand_placeholders, expand_setting, expand_variables
from tslab.study.loading import load_study
from tslab.study.settings import check_paths, from_presets, Settings, settings_to_args

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'Grid',
    'Override',
    'Settings',
    'Study',
    'StudyRun',
    'check_paths',
    'expand_placeholders',
    'expand_runs',
    'expand_setting',
    'expand_variables',
    'from_presets',
    'load_study',
    'match_problems',
    'merge_overrides',
    'run_matches',
    'select_runs',
    'setting_axis',
    'settings_to_args',
]
