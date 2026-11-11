"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.

This package exports the authoring surface, what a study module imports to
define its grids, and ``slurm_output`` for notebooks. The commands, machine
profiles and job records live in the submodules (``tslab.lab.study.commands``,
``tslab.lab.study.context``, ...).
"""

from tslab.lab.study.definition.after import After
from tslab.lab.study.definition.definition import (
    DEFAULT_RUN_SETTINGS,
    Grid,
    Override,
    setting_axis,
    Study,
    StudyRun,
)
from tslab.lab.study.definition.settings import (
    check_paths,
    from_presets,
    Settings,
    settings_to_args,
    SettingValue,
)
from tslab.lab.study.slurm import slurm_output

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
