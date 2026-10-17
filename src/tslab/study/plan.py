"""Resolving runs on a machine: settings and resources, merged per run."""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import logging

from tslab.study.definition import (
    Grid,
    match_problems,
    merge_overrides,
    run_matches,
    Study,
    StudyRun,
)
from tslab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.study.settings import check_paths, SettingValue

__all__ = ['ResolvedRun', 'resolve_runs']

_LOGGER = logging.getLogger('tslab.study')


@dataclass(frozen=True, kw_only=True)
class ResolvedRun:
    """One run with its machine's settings and resources applied.

    Args:
        run: The study run this resolves.
        settings: The run's settings, then the machine settings, then the
            matching machine overrides' settings; not yet expanded.
        slurm: The base ``[slurm]`` table with the matching overrides'
            ``slurm`` applied field by field, or ``None`` off SLURM.
    """

    run: StudyRun
    settings: Mapping[str, SettingValue]
    slurm: SlurmSettings | None


def resolve_runs(
    *, study: Study, machine: MachineProfile, runs: Sequence[StudyRun]
) -> list[ResolvedRun]:
    """Every run with the machine's settings and resources applied.

    Machine settings paths are checked against every grid they can apply to;
    an override that names an axis, grid or value this study lacks is logged
    once as a warning and still applies wherever it matches.

    Args:
        study: The study the runs came from.
        machine: The machine the runs run on.
        runs: The runs to resolve, e.g. ``expand_runs``' output.

    Returns:
        The resolved runs, in the order of ``runs``.

    Raises:
        ValueError: If a machine setting path is not a field of a grid it
            applies to; the error names the profile file, the override index
            or ``settings``, the grid and the paths.
    """
    _settings_problems(study=study, machine=machine)
    _override_warnings(study=study, machine=machine)
    return [
        ResolvedRun(
            run=run,
            settings=merge_overrides(
                base={**run.settings, **machine.settings},
                overrides=[(override.match, override.settings) for override in machine.overrides],
                grid_name=run.grid,
                values=run.values,
            ),
            slurm=_slurm_for(
                base=machine.slurm,
                overrides=machine.overrides,
                grid_name=run.grid,
                values=run.values,
            ),
        )
        for run in runs
    ]


def _settings_problems(*, study: Study, machine: MachineProfile) -> None:
    for grid_name, grid in study.grids.items():
        bad = check_paths(config_class=grid.config, settings=machine.settings)
        if bad:
            message = (
                f'{machine.path}: settings set paths that are not fields of '
                f'{grid.config.__name__} (grid {grid_name!r}): {", ".join(bad)}'
            )
            raise ValueError(message)
    for position, override in enumerate(machine.overrides):
        for grid_name, grid in _override_grids(
            study=study, grid_names=override.match.get('grid', ()), match=override.match
        ):
            bad = check_paths(config_class=grid.config, settings=override.settings)
            if bad:
                message = (
                    f'{machine.path}: override {position} sets paths that are not fields '
                    f'of {grid.config.__name__} (grid {grid_name!r}): {", ".join(bad)}'
                )
                raise ValueError(message)


def _override_grids(
    *, study: Study, grid_names: Sequence[str], match: Mapping[str, object]
) -> list[tuple[str, Grid]]:
    """The grids an override's settings apply to, for path checking."""
    axes = [key for key in match if key != 'grid']
    return [
        (name, grid)
        for name, grid in study.grids.items()
        if (not grid_names or name in grid_names) and all(axis in grid.axes for axis in axes)
    ]


def _override_warnings(*, study: Study, machine: MachineProfile) -> None:
    for position, override in enumerate(machine.overrides):
        problems = match_problems(grids=study.grids, match=override.match)
        if problems:
            _LOGGER.warning(
                'machine override %d of %s matches nothing in this study: %s',
                position,
                machine.path,
                '; '.join(problems),
            )


def _slurm_for(
    *,
    base: SlurmSettings | None,
    overrides: Sequence[MachineOverride],
    grid_name: str,
    values: Mapping[str, str],
) -> SlurmSettings | None:
    merged: dict[str, object] = {}
    if base is not None:
        merged = {item.name: getattr(base, item.name) for item in dataclasses.fields(base)}
    matching = [
        override
        for override in overrides
        if run_matches(grid_name=grid_name, values=values, match=override.match)
    ]
    matching.sort(key=lambda override: len(override.match))
    for override in matching:
        merged.update(override.slurm)
    # ty: ignore[invalid-argument-type] — merged holds SlurmSettings fields by name
    return SlurmSettings(**merged) if merged else None
