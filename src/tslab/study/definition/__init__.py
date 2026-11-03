"""The study definition: a round of experiments as grids of typed settings.

The package holds the ``Study`` and ``StudyRun`` types and their expansion
(``definition``), the string expansion of variables and placeholders
(``expansion``), the loader of a ``module:attribute`` reference (``loading``)
and the typed dotted settings they are built from (``settings``).
"""

from tslab.study.definition.definition import (
    DEFAULT_RUN_SETTINGS,
    expand_runs,
    followup_job_name,
    Grid,
    Match,
    match_problems,
    matching_in_order,
    merge_overrides,
    merge_settings,
    normalize_match,
    Override,
    RESERVED_AXIS_NAMES,
    RETRY_JOB,
    run_matches,
    select_runs,
    setting_axis,
    Study,
    study_job_names,
    StudyRun,
    WANDB_SYNC_JOB,
)
from tslab.study.definition.expansion import (
    expand_placeholders,
    expand_setting,
    expand_variables,
    PLACEHOLDERS,
)
from tslab.study.definition.loading import load_study
from tslab.study.definition.settings import (
    check_paths,
    field_annotation,
    from_presets,
    Settings,
    settings_to_args,
    SettingValue,
)

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'PLACEHOLDERS',
    'RESERVED_AXIS_NAMES',
    'RETRY_JOB',
    'WANDB_SYNC_JOB',
    'Grid',
    'Match',
    'Override',
    'SettingValue',
    'Settings',
    'Study',
    'StudyRun',
    'check_paths',
    'expand_placeholders',
    'expand_runs',
    'expand_setting',
    'expand_variables',
    'field_annotation',
    'followup_job_name',
    'from_presets',
    'load_study',
    'match_problems',
    'matching_in_order',
    'merge_overrides',
    'merge_settings',
    'normalize_match',
    'run_matches',
    'select_runs',
    'setting_axis',
    'settings_to_args',
    'study_job_names',
]
