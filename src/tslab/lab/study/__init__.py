"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the user's own ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.

This package exports the authoring surface, what a study module imports to
define its grids, and ``slurm_output`` for notebooks. The commands, machine
profiles and job records live in the submodules (``tslab.study.commands``,
``tslab.study.context``, ...).
"""

from tslab.study.definition import (
    After,
    check_paths,
    DEFAULT_RUN_SETTINGS,
    from_presets,
    Grid,
    Override,
    setting_axis,
    Settings,
    settings_to_args,
    SettingValue,
    Study,
    StudyRun,
)
from tslab.study.slurm import slurm_output

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'After',
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
    'slurm_output',
]
