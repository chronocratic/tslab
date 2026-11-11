"""Studies: named grids with the names, paths and settings checked at construction."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from tslab.lab.core.paths import validate_name_part
from tslab.lab.study.definition.grid import DEFAULT_RUN_SETTINGS, Grid, validate_name, value_kind
from tslab.lab.study.definition.matching import Match, match_problems, normalize_match
from tslab.lab.study.definition.runs import grid_runs, merge_settings, StudyRun
from tslab.lab.study.definition.settings import check_paths, Settings, SettingValue

if TYPE_CHECKING:
    from tslab.lab.study.reporting.result_table import ResultRow

__all__ = [
    'FOLLOWUP_KINDS',
    'RETRY_JOB',
    'WANDB_SYNC_JOB',
    'Study',
    'followup_job_name',
    'study_job_names',
]


#: The kinds of follow-up job; their SLURM names are ``<study>_<kind>``.
RETRY_JOB = 'retry'


WANDB_SYNC_JOB = 'wbsync'


@dataclass(frozen=True, kw_only=True)
class Study:
    """A named round of experiments with settings shared across its grids.

    Attributes:
        name: The study name.
        grids: The grid name to grid mapping.
        settings: Settings applied to every grid that has each path.
    """

    name: str
    grids: Mapping[str, Grid]
    settings: Settings = MappingProxyType({})

    def __post_init__(self) -> None:
        """Freeze and validate settings, grids, names and run paths."""
        if not isinstance(self.settings, Mapping):
            message = f'study settings must be a mapping, got {value_kind(value=self.settings)}'
            raise TypeError(message)
        object.__setattr__(self, 'settings', MappingProxyType(dict(self.settings)))
        validate_name(name=self.name, kind='study')
        if not self.grids:
            message = 'study must have at least one grid'
            raise ValueError(message)
        short_names: dict[str, str] = {}
        for grid_name, grid in self.grids.items():
            validate_name(name=grid_name, kind='grid')
            if not isinstance(grid, Grid):
                message = f'grid {grid_name!r} must be a Grid, got {value_kind(value=grid)}'
                raise TypeError(message)
            matches = (*grid.exclude, *(override.match for override in grid.overrides))
            problems = _grid_match_problems(grid_name=grid_name, grid=grid, matches=matches)
            if problems:
                message = f'grid {grid_name!r}: ' + '; '.join(problems)
                raise ValueError(message)
            if grid.short_name in short_names:
                message = (
                    f'grid {grid_name!r} reuses short_name {grid.short_name!r} '
                    f'of grid {short_names[grid.short_name]!r}'
                )
                raise ValueError(message)
            short_names[grid.short_name] = grid_name
            _grid_settings_problems(grid_name=grid_name, grid=grid)
            if next(grid_runs(grid_name=grid_name, grid=grid), None) is None:
                message = f'grid {grid_name!r} has no runs; every run is excluded'
                raise ValueError(message)
        self._check_settings()
        self._check_job_names()
        self._check_run_paths()
        from tslab.lab.study.definition.after import after_problems  # noqa: PLC0415 # why: breaks the import cycle

        problems = after_problems(study=self)
        if problems:
            message = '; '.join(problems)
            raise ValueError(message)

    def _check_settings(self) -> None:
        """Check that study paths exist and do not duplicate axes or run settings."""
        for path, value in self.settings.items():
            if any(
                not check_paths(config_class=grid.config, settings={path: value})
                for grid in self.grids.values()
            ):
                continue
            message = f'study {self.name!r} settings: {path} is no field of any grid'
            raise ValueError(message)
        problems = [
            problem
            for grid_name, grid in self.grids.items()
            for problem in _study_setting_conflicts(study=self, grid_name=grid_name, grid=grid)
        ]
        if problems:
            message = '; '.join(problems)
            raise ValueError(message)

    def _check_run_paths(self) -> None:
        """Raise unless every run path is made of valid names and no two runs share one."""
        owners: dict[str, tuple[str, dict[str, str]]] = {}
        descendants: dict[str, tuple[str, tuple[str, dict[str, str]]]] = {}
        runs = [
            (grid_name, grid, values)
            for grid_name, grid in self.grids.items()
            for values in grid_runs(grid_name=grid_name, grid=grid)
        ]
        for grid_name, grid, values in runs:
            text = grid.run_path_text(grid=grid_name, values=values)
            for part in text.split('/'):
                validate_name(name=part, kind=f'grid {grid_name!r} run path {text!r} part')
                validate_name_part(part=part)
            key = text.lower()  # a case-insensitive file system folds Coffee and coffee
            parts = key.split('/')
            prefixes = tuple('/'.join(parts[:index]) for index in range(1, len(parts)))
            if key in owners:
                other_grid, other_values = owners[key]
                message = (
                    f'grid {grid_name!r} run {values} has run path {text!r}, which '
                    f'grid {other_grid!r} run {other_values} has too; '
                    f'name more axes in run_path'
                )
                raise ValueError(message)
            nested = next(
                ((prefix, owners[prefix]) for prefix in prefixes if prefix in owners), None
            )
            if nested is None:
                nested = descendants.get(key)
            if nested is not None:
                owner_text, (other_grid, other_values) = nested
                message = (
                    f'grid {grid_name!r} run {values} has run path {text!r}, which is inside '
                    f'or around run path {owner_text!r} of grid {other_grid!r} run '
                    f'{other_values}; name more axes in run_path'
                )
                raise ValueError(message)
            owner = (grid_name, values)
            owners[key] = owner
            for prefix in (*prefixes, key):
                descendants.setdefault(prefix, (key, owner))

    def _check_job_names(self) -> None:
        """Raise when two grids give one job name, or a grid gives a follow-up name."""
        followups = {followup_job_name(study=self.name, kind=kind) for kind in FOLLOWUP_KINDS}
        owners: dict[str, str] = {}
        pairs = [
            (grid_name, name)
            for grid_name, names in study_job_names(study=self).items()
            for name in sorted(names)
        ]
        for grid_name, name in pairs:
            if name in followups:
                message = (
                    f'grid {grid_name!r} gives job name {name!r}, the follow-up job name {name!r}'
                )
                raise ValueError(message)
            if owners.setdefault(name, grid_name) != grid_name:
                message = (
                    f'grid {grid_name!r} gives job name {name!r}, '
                    f'which grid {owners[name]!r} gives too'
                )
                raise ValueError(message)

    def job_name_of(self, *, run: 'StudyRun') -> str:
        """The job name of one run: its grid ``job_name`` template filled in."""
        return self.grids[run.grid].job_name_for(study=self.name, values=run.values)

    def results(self, *, machine: Path, environment: Mapping[str, str]) -> 'list[ResultRow]':
        """The rows of the results table of every run of the study.

        Args:
            machine: The machine profile the runs run on.
            environment: The process environment used by the machine profile.

        Returns:
            One row per (run, instance), and one row per run without
            instances; the runs of every grid, in expansion order.
        """
        # why: imported here, not at the top, because the result table imports this module
        from tslab.lab.study.reporting.result_table import study_results  # noqa: PLC0415

        return study_results(study=self, machine=machine, environment=environment)


FOLLOWUP_KINDS = (RETRY_JOB, WANDB_SYNC_JOB)


def followup_job_name(*, study: str, kind: str) -> str:
    """The SLURM job name of a follow-up job of the study."""
    return f'{study}_{kind}'


def study_job_names(*, study: Study) -> dict[str, frozenset[str]]:
    """Per grid name, the job names its runs get; excluded runs are left out."""
    return {
        grid_name: frozenset(
            grid.job_name_for(study=study.name, values=values)
            for values in grid_runs(grid_name=grid_name, grid=grid)
        )
        for grid_name, grid in study.grids.items()
    }


def _grid_match_problems(*, grid_name: str, grid: Grid, matches: Sequence[Match]) -> list[str]:
    """The problems of the matches written inside one grid: a match there is about its own grid."""
    problems: list[str] = []
    for match in matches:
        normalized = normalize_match(match=match)
        foreign = [name for name in normalized.get('grid', ()) if name != grid_name]
        if foreign:
            problems.append(
                f'a match inside grid {grid_name!r} applies to its own grid only, '
                f'but names grid {", ".join(repr(name) for name in foreign)}'
            )
        rest = {key: value for key, value in normalized.items() if key != 'grid'}
        problems += match_problems(grids={grid_name: grid}, match=rest)
    return problems


def _study_setting_conflicts(*, study: Study, grid_name: str, grid: Grid) -> list[str]:
    """The axis and run-setting clashes of the study settings for one grid."""
    return [
        problem
        for study_path in study.settings
        for problem in _study_path_conflicts(
            study=study, grid_name=grid_name, grid=grid, study_path=study_path
        )
    ]


def _study_path_conflicts(
    *, study: Study, grid_name: str, grid: Grid, study_path: str
) -> list[str]:
    """The axis and run-setting clashes for one study setting path."""
    axis_conflicts = [
        f'study {study.name!r} sets {study_path}, also set by grid '
        f'{grid_name!r}, axis {axis_name!r}, value {value_name!r}; '
        f'keep it in one place, e.g. drop it from the axis with '
        f'without=({study_path!r},)'
        for axis_name, values in grid.axes.items()
        for value_name, axis_settings in values.items()
        if any(
            _setting_paths_overlap(first=study_path, second=axis_path)
            for axis_path in axis_settings
        )
    ]
    run_settings = merge_settings(base=DEFAULT_RUN_SETTINGS, update=grid.run_settings)
    if not any(
        _setting_paths_overlap(first=study_path, second=run_path) for run_path in run_settings
    ):
        return axis_conflicts
    message = (
        f'study {study.name!r} sets {study_path}, also set by grid '
        f'{grid_name!r}, run_settings; keep it in one place'
    )
    return [*axis_conflicts, message]


def _setting_paths_overlap(*, first: str, second: str) -> bool:
    """Whether one dotted setting path equals or contains the other."""
    return first == second or first.startswith(f'{second}.') or second.startswith(f'{first}.')


def _grid_settings_problems(*, grid_name: str, grid: Grid) -> None:
    """Name the grid, the axis and value, or run_settings or override N, for each bad path."""
    for axis_name, values in grid.axes.items():
        for value_name, settings in values.items():
            _check_setting_paths(
                grid=grid,
                settings=settings,
                where=f'grid {grid_name!r}: axis {axis_name!r} value {value_name!r} sets',
            )
    _check_setting_paths(
        grid=grid, settings=grid.run_settings, where=f'grid {grid_name!r}: run_settings set'
    )
    for position, override in enumerate(grid.overrides):
        _check_setting_paths(
            grid=grid,
            settings=override.settings,
            where=f'grid {grid_name!r}: override {position} sets',
        )


def _check_setting_paths(*, grid: Grid, settings: Mapping[str, SettingValue], where: str) -> None:
    """Raise unless every path of ``settings`` is a field of the config of the grid."""
    bad = check_paths(config_class=grid.config, settings=settings)
    if bad:
        message = f'{where} paths that are not fields of {grid.config.__name__}: {", ".join(bad)}'
        raise ValueError(message)
