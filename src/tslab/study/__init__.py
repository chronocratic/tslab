"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the user's own ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.

This package exports the authoring surface only: what a study module imports
to define its grids. The commands, machine profiles and job records live in the
submodules (``tslab.study.commands``, ``tslab.study.context``, ...).
"""

from tslab.study.definition import (
    DEFAULT_RUN_SETTINGS,
    Grid,
    Override,
    setting_axis,
    Study,
    StudyRun,
)
from tslab.study.settings import check_paths, from_presets, Settings, settings_to_args, SettingValue

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'Grid',
    'Override',
    'SettingValue',
    'Settings',
    'Study',
    'StudyRun',
    'check_paths',
    'from_presets',
    'setting_axis',
    'settings_to_args',
]
