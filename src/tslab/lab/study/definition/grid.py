"""Grids: one script command with named axes of settings, and the overrides that refine runs."""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import TYPE_CHECKING

from tslab.lab.core.paths import NAME_PART_PATTERN
from tslab.lab.study.definition.columns import (
    ERROR_COLUMN,
    GRID_COLUMN,
    INSTANCE_COLUMN,
    INSTANCE_STATE_COLUMN,
    RUN_PATH_COLUMN,
    SECONDS_COLUMN,
    STUDY_STATE_COLUMN,
)
from tslab.lab.study.definition.expansion import PLACEHOLDERS
from tslab.lab.study.definition.matching import Match
from tslab.lab.study.definition.settings import Settings, SettingValue

if TYPE_CHECKING:
    from tslab.lab.study.definition.after import After

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'JOB_NAME_PLACEHOLDER_AXES',
    'RESERVED_AXIS_NAMES',
    'Grid',
    'Override',
    'setting_axis',
    'validate_name',
    'value_kind',
]


_SHORT_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-]+')


_JOB_NAME_PLACEHOLDER = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')


_JOB_NAME_LITERAL = re.compile(r'[A-Za-z0-9._-]*')


_RUN_PATH_LITERAL = re.compile(r'[A-Za-z0-9._/-]*')


_TEMPLATE_HINTS = {
    'grid job_name': 'short_name, study or an axis',
    'grid run_path': 'grid, short_name or an axis',
}


#: Names a job name template can use besides the axes; no axis may take one.
JOB_NAME_PLACEHOLDER_AXES = frozenset({'short_name', *PLACEHOLDERS})


#: Axis names that are columns of the results table or the match key of a grid.
RESERVED_AXIS_NAMES = frozenset(
    {
        GRID_COLUMN,
        RUN_PATH_COLUMN,
        STUDY_STATE_COLUMN,
        INSTANCE_COLUMN,
        INSTANCE_STATE_COLUMN,
        SECONDS_COLUMN,
        ERROR_COLUMN,
    }
)


DEFAULT_RUN_SETTINGS: Mapping[str, str] = MappingProxyType(
    {'output_root': '{output_root}/{study}', 'run_path': '{run_path}'}
)


def validate_name(*, name: str, kind: str) -> str:
    """Check a study, grid, axis or value name; return it when valid.

    Args:
        name: The name to check.
        kind: What the name is, for the error message, e.g. ``'grid'``.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If the name is empty, starts with a dot (the hidden study
            folder is ``.study``), or holds any character outside
            ``[A-Za-z0-9._-]``.
    """
    if not name or name.startswith('.') or NAME_PART_PATTERN.fullmatch(name) is None:
        message = (
            f'invalid {kind} name {name!r}; names use letters, digits, dot, dash or underscore, '
            f'and do not start with a dot'
        )
        raise ValueError(message)
    return name


def setting_axis(
    *, path: str, values: Sequence[SettingValue]
) -> dict[str, dict[str, SettingValue]]:
    """An axis that varies one setting over the given values.

    Args:
        path: The dotted field path to vary.
        values: The values of the setting, in the order of the axis.

    Returns:
        ``{str(value): {path: value}}`` in the order of ``values``.

    Raises:
        ValueError: If two values have the same text, which would be one
            value name.
    """
    names = [str(value) for value in values]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        message = f'setting_axis({path!r}): the value {repeated[0]!r} is given twice'
        raise ValueError(message)
    return {str(value): {path: value} for value in values}


@dataclass(frozen=True, kw_only=True)
class Override:
    """Settings applied to the runs whose axis values match.

    Attributes:
        match: The axis names and the values that select the runs.
        settings: The settings the selected runs get.
    """

    match: Match
    settings: Settings

    def __post_init__(self) -> None:
        """Check that match and settings are mappings."""
        for name in ('match', 'settings'):
            if not isinstance(getattr(self, name), Mapping):
                message = (
                    f'Override {name} must be a mapping, got {type(getattr(self, name)).__name__}'
                )
                raise TypeError(message)


@dataclass(frozen=True, kw_only=True)
class Grid:
    """One script command of a study, run over the product of its axes.

    Attributes:
        short_name: Unique in the study; the default job name and the stem of
            the task log files.
        command: The no-preset command of the script, as a word tuple.
        config: The config class of the script that the settings are typed against.
        axes: Axis name to ``{value name: settings}``, in run-path order.
        exclude: Matches naming the runs to drop.
        overrides: Settings applied to matching runs.
        run_settings: Settings every run of the grid gets, added to the default
            run settings. Read-only once built.
        job_name: The job name as a template of ``{short_name}``,
            ``{study}`` and axis names; ``{study}_{short_name}`` by default.
            Runs with different names go into different arrays, so naming an
            axis splits the grid by it.
        run_path: The run path as a template of ``{grid}``, ``{short_name}``
            and axis names; ``None`` is the grid then the axis values in axis
            order.
        after: Upstream grids that each run must wait for.
    """

    short_name: str
    command: tuple[str, ...]
    config: type
    axes: Mapping[str, Mapping[str, Settings]]
    exclude: tuple[Match, ...] = ()
    overrides: tuple[Override, ...] = ()
    run_settings: Settings = DEFAULT_RUN_SETTINGS
    job_name: str = '{study}_{short_name}'
    run_path: str | None = None
    after: tuple['After', ...] = ()

    @property
    def job_name_axes(self) -> tuple[str, ...]:
        """The axes the job name template uses, in template order."""
        names = [str(match[1]) for match in _JOB_NAME_PLACEHOLDER.finditer(self.job_name)]
        return tuple(name for name in dict.fromkeys(names) if name in self.axes)

    def job_name_for(self, *, study: str, values: Mapping[str, str]) -> str:
        """The job name of one run.

        Args:
            study: The study name.
            values: The axis values of the run, as ``StudyRun.values``.

        Returns:
            The template with its placeholders filled in.
        """
        known = {'short_name': self.short_name, 'study': study, **values}
        return _JOB_NAME_PLACEHOLDER.sub(lambda match: known[match.group(1)], self.job_name)

    @property
    def run_path_template(self) -> str:
        """The run path template: the given one, or the grid then the axis values."""
        if self.run_path is not None:
            return self.run_path
        return '/'.join(['{grid}', *(f'{{{axis}}}' for axis in self.axes)])

    def run_path_text(self, *, grid: str, values: Mapping[str, str]) -> str:
        """The run path of one run as text, with empty parts kept for the checks.

        Args:
            grid: The grid name.
            values: The axis values of the run, as ``StudyRun.values``.

        Returns:
            The template with its placeholders filled in.
        """
        known = {'grid': grid, 'short_name': self.short_name, **values}
        return _JOB_NAME_PLACEHOLDER.sub(
            lambda match: known[match.group(1)], self.run_path_template
        )

    def run_path_for(self, *, grid: str, values: Mapping[str, str]) -> PurePosixPath:
        """The run path of one run.

        Args:
            grid: The grid name.
            values: The axis values of the run, as ``StudyRun.values``.

        Returns:
            The filled template as a path.
        """
        return PurePosixPath(self.run_path_text(grid=grid, values=values))

    def _check_job_name(self) -> None:
        """Raise unless the template has plain characters and known placeholders."""
        if not self.job_name:
            message = 'grid job_name must not be empty'
            raise ValueError(message)
        _check_template(
            label='grid job_name',
            template=self.job_name,
            literal=_JOB_NAME_LITERAL,
            literal_text='letters, digits, dot, dash or underscore',
            allowed={'short_name', 'study', *self.axes},
        )

    def _check_run_path(self) -> None:
        """Raise unless the run path template has plain characters and known placeholders."""
        _check_template(
            label='grid run_path',
            template=self.run_path_template,
            literal=_RUN_PATH_LITERAL,
            literal_text='letters, digits, dot, dash, underscore or slash',
            allowed={'grid', 'short_name', *self.axes},
        )

    def __post_init__(self) -> None:
        """Check the shape: names, command, config, axes."""
        if _SHORT_NAME_PATTERN.fullmatch(self.short_name) is None:
            message = (
                f'invalid short_name {self.short_name!r}; '
                f'use letters, digits, dashes or underscores'
            )
            raise ValueError(message)
        if not self.command:
            message = 'grid command must not be empty'
            raise ValueError(message)
        if not (isinstance(self.config, type) and dataclasses.is_dataclass(self.config)):
            message = f'grid config must be a dataclass, got {type(self.config).__qualname__}'
            raise ValueError(message)
        self._check_shapes()
        if not self.axes:
            message = 'grid must have at least one axis'
            raise ValueError(message)
        for axis_name, values in self.axes.items():
            validate_name(name=axis_name, kind='axis')
            if axis_name in RESERVED_AXIS_NAMES | JOB_NAME_PLACEHOLDER_AXES:
                reserved = RESERVED_AXIS_NAMES | JOB_NAME_PLACEHOLDER_AXES
                message = (
                    f'axis name {axis_name!r} is reserved (a results column, the grid key '
                    f'or a placeholder); reserved: {", ".join(sorted(reserved))}'
                )
                raise ValueError(message)
            if not values:
                message = f'axis {axis_name!r} must have at least one value'
                raise ValueError(message)
            for value_name in values:
                validate_name(name=value_name, kind=f'axis {axis_name!r} value')
        self._check_job_name()
        self._check_run_path()

    def _check_shapes(self) -> None:
        """Type-check the containers; the usual slip is a missing trailing comma."""
        self._check_axis_shapes()
        self._check_after_shapes()
        if isinstance(self.exclude, Mapping):
            message = 'grid exclude must be a tuple of matches; trailing comma: exclude=({...},)?'
            raise TypeError(message)
        for match in self.exclude:
            if not isinstance(match, Mapping):
                message = (
                    f'each grid exclude must be a match mapping, got {value_kind(value=match)}'
                )
                raise TypeError(message)
        if isinstance(self.overrides, Override):
            message = (
                'grid overrides must be a tuple of Override; '
                'trailing comma: overrides=(Override(...),)?'
            )
            raise TypeError(message)
        for override in self.overrides:
            if not isinstance(override, Override):
                message = (
                    f'each grid override must be an Override, got {value_kind(value=override)}'
                )
                raise TypeError(message)
        if not isinstance(self.run_settings, Mapping):
            message = (
                f'grid run_settings must be a mapping, got {value_kind(value=self.run_settings)}'
            )
            raise TypeError(message)

    def _check_axis_shapes(self) -> None:
        if not isinstance(self.axes, Mapping):
            message = (
                'grid axes must be a mapping of axis name to values, '
                f'got {value_kind(value=self.axes)}'
            )
            raise TypeError(message)
        for axis_name, values in self.axes.items():
            _check_axis_values(axis_name=axis_name, values=values)

    def _check_after_shapes(self) -> None:
        """Type-check the dependency tuple; the usual slip is a missing trailing comma."""
        from tslab.lab.study.definition.after import After  # noqa: PLC0415 # why: breaks the import cycle

        if isinstance(self.after, After):
            message = 'grid after must be a tuple of After; trailing comma: after=(After(...),)?'
            raise TypeError(message)
        if not isinstance(self.after, tuple):
            message = f'grid after must be a tuple of After, got {value_kind(value=self.after)}'
            raise TypeError(message)
        for after in self.after:
            if not isinstance(after, After):
                message = f'each grid after must be an After, got {value_kind(value=after)}'
                raise TypeError(message)


def _check_axis_values(*, axis_name: str, values: object) -> None:
    """Raise unless the axis maps value names to settings mappings."""
    if not isinstance(values, Mapping):
        message = (
            f'axis {axis_name!r} must map value names to settings, '
            f'got {value_kind(value=values)}; for one setting use '
            f'setting_axis(path=..., values=...)'
        )
        raise TypeError(message)
    for value_name, settings in values.items():
        if not isinstance(settings, Mapping):
            message = (
                f'axis {axis_name!r} value {value_name!r} must be a mapping of '
                f'setting path to value, got {value_kind(value=settings)}'
            )
            raise TypeError(message)


def value_kind(*, value: object) -> str:
    """The type name of ``value``, for error messages."""
    return type(value).__name__


def _check_template(
    *, label: str, template: str, literal: re.Pattern[str], literal_text: str, allowed: set[str]
) -> None:
    """Raise unless the template has only plain characters and allowed placeholders."""
    for piece in _JOB_NAME_PLACEHOLDER.split(template)[::2]:
        if literal.fullmatch(piece) is None:
            message = (
                f'{label} {template!r}: {piece!r} is not allowed; use {literal_text} '
                f'between placeholders'
            )
            raise ValueError(message)
    for name in _JOB_NAME_PLACEHOLDER.findall(template):
        if name not in allowed:
            message = (
                f'{label} {template!r}: unknown placeholder {name!r}; '
                f'use {_TEMPLATE_HINTS[label]}: {", ".join(sorted(allowed))}'
            )
            raise ValueError(message)
