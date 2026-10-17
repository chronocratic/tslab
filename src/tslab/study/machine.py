"""Machine profiles: one TOML file per machine, read strictly.

A profile names the launcher, the output root, settings and environment for
every run on that machine, the SLURM resources and overrides matched by axis
values. Unknown keys raise: a typo in a profile must fail at submit, not
silently change a cluster job.
"""

import dataclasses
from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import cast, Literal

from tslab.study.expansion import expand_variables
from tslab.study.settings import SettingValue

__all__ = [
    'FollowupSettings',
    'MachineOverride',
    'MachineProfile',
    'MachineProfileError',
    'SlurmSettings',
    'flatten_settings',
    'load_machine',
]

_TOP_LEVEL_KEYS = frozenset(
    {
        'launcher',
        'output_root',
        'tslab_command',
        'settings',
        'env',
        'slurm',
        'followup',
        'overrides',
    }
)


class MachineProfileError(ValueError):
    """The profile file does not hold a machine profile."""


@dataclass(frozen=True, kw_only=True)
class SlurmSettings:
    """The ``[slurm]`` table: the resources of one sbatch job."""

    partition: str
    time: str
    qos: str | None = None
    cpus_per_task: int = 1
    mem_per_cpu: str | None = None
    gpus: int = 0
    max_concurrent: int = 100
    max_array_size: int = 1000
    sbatch_extra: tuple[str, ...] = ()
    node_local_tmp: bool = True
    submit_pause_seconds: float = 0.5


@dataclass(frozen=True, kw_only=True)
class FollowupSettings:
    """The ``[followup]`` table: the small helper jobs of a study."""

    auto_retry: bool = False
    max_attempts: int = 3
    wandb_sync: bool = False
    time: str = '00:30:00'


@dataclass(frozen=True, kw_only=True)
class MachineOverride:
    """One ``[[overrides]]`` entry: match plus slurm and/or settings."""

    match: dict[str, tuple[str, ...]]
    slurm: dict[str, object] = field(default_factory=dict)
    settings: dict[str, SettingValue] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MachineProfile:
    """A machine profile read from TOML, fully checked."""

    path: Path
    launcher: Literal['slurm', 'local']
    output_root: Path
    tslab_command: tuple[str, ...] = ('uv', 'run', 'tslab')
    settings: dict[str, SettingValue] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    slurm: SlurmSettings | None = None
    followup: FollowupSettings = FollowupSettings()
    overrides: tuple[MachineOverride, ...] = ()


def load_machine(*, path: Path) -> MachineProfile:
    """The machine profile a TOML file holds, checked key by key.

    Args:
        path: The profile file to read.

    Returns:
        The profile, with ``output_root`` and the ``[env]`` values expanded
        from the process environment.

    Raises:
        MachineProfileError: If the TOML does not parse, or holds an unknown
            or missing key, or a value of the wrong type; the error names
            the file and the location, e.g. ``slurm.cpu_per_task``.
    """
    try:
        table = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        message = f'{path}: the TOML does not parse: {error}'
        raise MachineProfileError(message) from error
    return _machine_profile(path=path, table=table)


def _machine_profile(*, path: Path, table: dict[str, object]) -> MachineProfile:
    unknown = [name for name in table if name not in _TOP_LEVEL_KEYS]
    if unknown:
        message = f'{path}: {unknown[0]} is not a known top-level key'
        raise MachineProfileError(message)
    launcher = _string_field(table=table, name='launcher', path=path)
    if launcher not in {'slurm', 'local'}:
        message = f'{path}: launcher must be "slurm" or "local", got {launcher!r}'
        raise MachineProfileError(message)
    output_root = _string_field(table=table, name='output_root', path=path)
    env = _expand_env(table=table, path=path)
    slurm = _from_table(
        cls=SlurmSettings,
        table=table.get('slurm'),
        location='slurm',
        path=path,
        required=launcher == 'slurm',
    )
    followup = _from_table(
        cls=FollowupSettings,
        table=table.get('followup'),
        location='followup',
        path=path,
        required=False,
    )
    overrides = tuple(
        override
        for override in (
            _from_table(
                cls=MachineOverride,
                table=entry,
                location=f'overrides[{position}]',
                path=path,
                required=True,
            )
            for position, entry in enumerate(_overrides_table(table=table, path=path))
        )
        if override is not None
    )
    return MachineProfile(
        path=path,
        launcher=launcher,
        output_root=Path(expand_variables(text=output_root, env={**os.environ, **env})),
        tslab_command=_string_tuple(
            table=table, name='tslab_command', path=path, default=('uv', 'run', 'tslab')
        ),
        settings=flatten_settings(table=_settings_table(table=table)),
        env=env,
        slurm=slurm,
        followup=followup or FollowupSettings(),
        overrides=overrides,
    )


def _string_field(*, table: dict[str, object], name: str, path: Path) -> str:
    if name not in table:
        message = f'{path}: {name} is required'
        raise MachineProfileError(message)
    value = table[name]
    if not isinstance(value, str):
        message = f'{path}: {name} must be a string, got {type(value).__name__}'
        raise MachineProfileError(message)
    return value


def _string_tuple(
    *, table: dict[str, object], name: str, path: Path, default: tuple[str, ...]
) -> tuple[str, ...]:
    value = table.get(name, list(default))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        message = f'{path}: {name} must be a list of strings'
        raise MachineProfileError(message)
    return tuple(cast('list[str]', value))


def _expand_env(*, table: dict[str, object], path: Path) -> dict[str, str]:
    value = table.get('env', {})
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        message = f'{path}: [env] must hold string values'
        raise MachineProfileError(message)
    expanded: dict[str, str] = {}
    for name, item in value.items():
        try:
            expanded[name] = expand_variables(text=item, env={**os.environ, **expanded})
        except ValueError as error:
            message = f'{path}: [env] value for {name!r}: {error}'
            raise MachineProfileError(message) from error
    return expanded


def _overrides_table(*, table: dict[str, object], path: Path) -> list[object]:
    value = table.get('overrides', [])
    if not isinstance(value, list):
        message = f'{path}: overrides must be an array of tables'
        raise MachineProfileError(message)
    return cast('list[object]', value)


def _settings_table(*, table: dict[str, object]) -> dict[str, object]:
    value = table.get('settings')
    return cast('dict[str, object]', value) if isinstance(value, dict) else {}


def _from_table[T](
    *, cls: type[T], table: object, location: str, path: Path, required: bool
) -> T | None:
    if table is None:
        if required:
            message = f'{path}: the {location} table is required'
            raise MachineProfileError(message)
        return None
    if not isinstance(table, dict):
        message = f'{path}: {location} must be a table, got {type(table).__name__}'
        raise MachineProfileError(message)
    values = _table_values(cls=cls, table=table, location=location, path=path)
    try:
        return cls(**values)  # type: ignore[call-arg]
    except TypeError as error:
        message = f'{path}: {location}: {error}'
        raise MachineProfileError(message) from error


def _table_values(
    *, cls: type, table: dict[str, object], location: str, path: Path
) -> dict[str, object]:
    fields = dataclasses.fields(cls)  # ty: ignore[invalid-argument-type]
    if cls is MachineOverride and 'match' not in table:
        message = f'{path}: {location}.match is required'
        raise MachineProfileError(message)
    known = {item.name for item in fields}
    unknown = [name for name in table if name not in known]
    if unknown:
        message = f'{path}: {location}.{unknown[0]} is not a known key of {cls.__name__}'
        raise MachineProfileError(message)
    values: dict[str, object] = {}
    for item in fields:
        if item.name not in table or (
            cls is MachineOverride and item.name in {'match', 'slurm', 'settings'}
        ):
            continue
        values[item.name] = _check_value(
            annotation=item.type,
            value=table[item.name],
            location=f'{location}.{item.name}',
            path=path,
        )
    if cls is MachineOverride:
        values['match'] = _match_value(value=table['match'], location=location, path=path)
        values['slurm'] = _slurm_overrides(
            table=table.get('slurm', {}), location=location, path=path
        )
        values['settings'] = flatten_settings(table=_settings_table(table=table))
        if not table.get('slurm') and not table.get('settings'):
            message = f'{path}: {location} needs a slurm or a settings table'
            raise MachineProfileError(message)
    return values


def _match_value(*, value: object, location: str, path: Path) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        message = f'{path}: {location}.match must be a non-empty table'
        raise MachineProfileError(message)
    match: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        names = [item] if isinstance(item, str) else item
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            message = f'{path}: {location}.match.{key} must be a name or a list of names'
            raise MachineProfileError(message)
        match[key] = tuple(names)
    return match


def _slurm_overrides(*, table: object, location: str, path: Path) -> dict[str, object]:
    if not isinstance(table, dict):
        message = f'{path}: {location}.slurm must be a table'
        raise MachineProfileError(message)
    fields = {item.name: item for item in dataclasses.fields(SlurmSettings)}
    unknown = [key for key in table if key not in fields]
    if unknown:
        message = f'{path}: {location}.slurm.{unknown[0]} is not a known key of SlurmSettings'
        raise MachineProfileError(message)
    values: dict[str, object] = {}
    for key, item in table.items():
        values[key] = _check_value(
            annotation=fields[key].type, value=item, location=f'{location}.slurm.{key}', path=path
        )
    return values


def _check_value(*, annotation: object, value: object, location: str, path: Path) -> object:
    # ponytail: dataclasses.fields gives real annotation objects for these
    # few forms (str, int, float, bool, str | None, tuple, dict); parse those,
    # not a whole typing engine.
    if 'tuple' in str(annotation) or 'dict' in str(annotation):
        return _container_value(annotation=annotation, value=value, location=location, path=path)
    return _scalar_value(annotation=annotation, value=value, location=location, path=path)


def _scalar_value(*, annotation: object, value: object, location: str, path: Path) -> object:
    if value is None:
        return _null_value(annotation=annotation, location=location, path=path)
    if isinstance(value, bool):
        if annotation is not bool:
            message = f'{path}: {location} must be {annotation}, got a bool'
            raise MachineProfileError(message)
        return value
    if annotation is int:
        if isinstance(value, int):
            return value
        message = f'{path}: {location} must be an integer, got {type(value).__name__}'
        raise MachineProfileError(message)
    if annotation is float:
        if isinstance(value, (int, float)):
            return value
        message = f'{path}: {location} must be a number, got {type(value).__name__}'
        raise MachineProfileError(message)
    if annotation is str or str(annotation) == 'str | None':
        if isinstance(value, str):
            return value
        message = f'{path}: {location} must be a string, got {type(value).__name__}'
        raise MachineProfileError(message)
    message = f'{path}: {location} has an unsupported type {annotation}'
    raise MachineProfileError(message)


def _null_value(*, annotation: object, location: str, path: Path) -> None:
    if 'NoneType' in str(annotation):
        return
    message = f'{path}: {location} must be {annotation}, got null'
    raise MachineProfileError(message)


def _container_value(*, annotation: object, value: object, location: str, path: Path) -> object:
    if 'dict' in str(annotation):
        if not isinstance(value, dict):
            message = f'{path}: {location} must be a table, got {type(value).__name__}'
            raise MachineProfileError(message)
        return value
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        message = f'{path}: {location} must be a list of strings'
        raise MachineProfileError(message)
    return tuple(value)


def flatten_settings(*, table: dict[str, object], prefix: str = '') -> dict[str, SettingValue]:
    """A settings table as dotted paths; lists become tuples.

    A nested table ``[settings.tracking]`` with ``wandb_mode = "offline"`` is
    the same as ``"tracking.wandb_mode" = "offline"``.

    Args:
        table: The table to flatten.
        prefix: The dotted prefix of this table, for recursion.

    Returns:
        ``{dotted path: value}`` with every list a tuple.
    """
    flat: dict[str, SettingValue] = {}
    for key, value in table.items():
        dotted = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_settings(table=value, prefix=dotted))
        else:
            flat[dotted] = tuple(value) if isinstance(value, list) else cast('SettingValue', value)
    return flat
