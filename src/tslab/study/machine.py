"""Machine profiles: one TOML file per machine, read strictly.

A profile names the launcher, the output root, settings and environment for
every run on that machine, the SLURM resources and overrides matched by axis
values. Unknown keys raise: a typo in a profile must fail at submit, not
silently change a cluster job.
"""

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tomllib
from typing import cast, Literal

from tslab.core.config import ConfigFileError, decode_dataclass
from tslab.core.paths import atomic_write_text
from tslab.helpers.text import unknown_key_message
from tslab.study.expansion import expand_variables
from tslab.study.settings import SettingValue

__all__ = [
    'FollowupSettings',
    'MachineLocation',
    'MachineOverride',
    'MachineProfile',
    'MachineProfileError',
    'SlurmSettings',
    'flatten_settings',
    'load_machine',
    'read_register',
    'register_file',
    'resolve_machine',
    'validate_machine_name',
    'write_register',
]

_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# why: the smallest value each numeric key accepts; the keys are unique across the tables
_MINIMUMS: dict[str, float] = {
    'cpus_per_task': 1,
    'gpus': 0,
    'signal_seconds': 0,
    'max_concurrent': 1,
    'max_array_size': 1,
    'submit_pause_seconds': 0,
    'max_attempts': 1,
}


class MachineProfileError(ValueError):
    """The profile file does not hold a machine profile."""


@dataclass(frozen=True, kw_only=True)
class SlurmSettings:
    """The ``[slurm]`` table: the resources and setup of one sbatch job.

    Attributes:
        partition: The required SLURM partition.
        time: The job time limit.
        qos: The quality of service, when set.
        cpus_per_task: CPUs allocated to each task.
        mem_per_cpu: Memory allocated to each CPU, when set.
        gpus: GPUs allocated to each task.
        gpu_type: The GPU type for the GRES request, when set.
        max_concurrent: The maximum number of concurrent array jobs.
        max_array_size: The maximum number of tasks in an array.
        sbatch_extra: Additional raw ``#SBATCH`` options.
        node_local_tmp: Whether to use node-local temporary folders.
        submit_pause_seconds: Delay between job submissions.
        setup: Shell lines run before each task.
        signal_seconds: Seconds before the time limit to send ``SIGUSR2``; zero disables it.
    """

    partition: str
    time: str = '2-00:00:00'
    qos: str | None = None
    cpus_per_task: int = 1
    mem_per_cpu: str | None = None
    gpus: int = 0
    gpu_type: str | None = None
    max_concurrent: int = 100
    max_array_size: int = 1000
    sbatch_extra: tuple[str, ...] = ()
    node_local_tmp: bool = True
    submit_pause_seconds: float = 0.5
    setup: tuple[str, ...] = ()
    signal_seconds: int = 120


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


_TOP_LEVEL_KEYS = frozenset(
    item.name for item in dataclasses.fields(MachineProfile) if item.name != 'path'
)


_REGISTER_VARIABLE = 'TSLAB_MACHINES_FILE'
_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-]+')


def register_file() -> Path:
    """The register of machine names: ``tslab machine add`` writes it.

    Returns:
        ``$TSLAB_MACHINES_FILE`` when set, else ``$XDG_CONFIG_HOME/tslab/machines.toml``,
        else ``~/.config/tslab/machines.toml``.
    """
    override = os.environ.get(_REGISTER_VARIABLE, '')
    if override != '':
        return Path(override)
    config = os.environ.get('XDG_CONFIG_HOME', '')
    base = Path(config) if config != '' else Path.home() / '.config'
    return base / 'tslab' / 'machines.toml'


def validate_machine_name(*, name: str) -> str:
    """Check that ``name`` is a plain stem: letters, digits, ``_`` and ``-``.

    Args:
        name: The name to register.

    Returns:
        ``name``, unchanged.

    Raises:
        MachineProfileError: If ``name`` is empty, has a separator, a dot or a space.
    """
    if not _NAME_PATTERN.fullmatch(name):
        message = (
            f'machine name {name!r} must be a plain stem of letters, digits, "_" and "-" '
            '(no path separator, no .toml); example: cluster'
        )
        raise MachineProfileError(message)
    return name


def read_register() -> dict[str, Path]:
    """The registered machine names and their profile files.

    Returns:
        The name to absolute path mapping; empty when the register does not exist.

    Raises:
        MachineProfileError: If the register is not a table of name to path strings.
    """
    path = register_file()
    if not path.is_file():
        return {}
    try:
        table = tomllib.loads(path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as error:
        message = f'machine register {path} is not valid TOML: {error}; fix or delete the file'
        raise MachineProfileError(message) from error
    if not all(isinstance(value, str) for value in table.values()):
        message = f'machine register {path} must map names to path strings; fix or delete the file'
        raise MachineProfileError(message)
    return {name: Path(cast('str', value)) for name, value in table.items()}


def write_register(*, entries: Mapping[str, Path]) -> None:
    """Replace the register with ``entries``, atomically.

    Args:
        entries: The name to absolute path mapping to store.
    """
    path = register_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{name} = {json.dumps(file.as_posix())}\n' for name, file in sorted(entries.items())]
    atomic_write_text(path=path, text=''.join(lines))


@dataclass(frozen=True, kw_only=True)
class MachineLocation:
    """The profile file a ``--machine`` value names.

    Attributes:
        path: The absolute path of the profile file.
        project_root: The folder the jobs run in: the folder that holds
            ``machines/`` when the name was found there, else the start folder.
    """

    path: Path
    project_root: Path


def resolve_machine(*, machine: str, start: Path) -> MachineLocation:
    """Find the profile file a ``--machine`` value names.

    A value with a path separator, a ``.toml`` suffix or that is an existing
    file is a path. Any other value is a name, looked up in this order:
    ``machines/<name>.toml`` in ``start`` and each folder above it (the first
    hit wins), then the register of ``tslab machine add``.

    Args:
        machine: The name or the path given on the command line.
        start: The folder to search from, usually the working directory.

    Returns:
        The absolute profile path and the folder the jobs run in.

    Raises:
        MachineProfileError: If a name is neither found above ``start`` nor
            registered, or is registered with a file that is gone.
    """
    if '/' in machine or '\\' in machine or machine.endswith('.toml') or Path(machine).is_file():
        return MachineLocation(path=Path(machine).resolve(), project_root=start)
    folders = [folder / 'machines' for folder in (start, *start.parents)]
    for folder in folders:
        if (folder / f'{machine}.toml').is_file():
            return MachineLocation(path=folder / f'{machine}.toml', project_root=folder.parent)
    registered = read_register()
    if machine in registered:
        path = registered[machine]
        if not path.is_file():
            message = (
                f'machine {machine!r} is registered with {path}, which does not exist; '
                f'register the new location with: tslab machine add {machine} <path>, '
                f'or drop it with: tslab machine remove {machine}'
            )
            raise MachineProfileError(message)
        return MachineLocation(path=path, project_root=start)
    message = (
        f'no machine named {machine!r}: no machines/{machine}.toml in {start} or a folder '
        f'above it, and the registered names are: {", ".join(sorted(registered)) or "none"}. '
        f'Pass a profile path, or register one with: tslab machine add {machine} <path>'
    )
    raise MachineProfileError(message)


def load_machine(*, path: Path, lenient_env: bool = False) -> MachineProfile:
    """The machine profile a TOML file holds, checked key by key.

    Args:
        path: The profile file to read.
        lenient_env: Keep an ``[env]`` value with an unset ``${VAR}`` as it is
            instead of raising, for commands that only read state and never
            launch a job (``output_root`` is always expanded strictly).

    Returns:
        The profile, with ``output_root`` and the ``[env]`` values expanded
        from the process environment.

    Raises:
        MachineProfileError: If the TOML does not parse, or holds an unknown
            or missing key, a value of the wrong type or out of range, the
            file cannot be read, or a SLURM table is given for a local
            launcher; the error names the file and the location, e.g.
            ``slurm.cpu_per_task``.
    """
    try:
        table = tomllib.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        message = f'{path}: the machine file cannot be read: {error}'
        raise MachineProfileError(message) from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        message = f'{path}: the TOML does not parse: {error}'
        raise MachineProfileError(message) from error
    return _machine_profile(path=path, table=table, lenient_env=lenient_env)


def _machine_profile(*, path: Path, table: dict[str, object], lenient_env: bool) -> MachineProfile:
    unknown = [name for name in table if name not in _TOP_LEVEL_KEYS]
    if unknown:
        problem = unknown_key_message(location='', key=unknown[0], valid=_TOP_LEVEL_KEYS)
        message = f'{path}: {problem}'
        raise MachineProfileError(message)
    launcher = _string_field(table=table, name='launcher', path=path)
    if launcher not in {'slurm', 'local'}:
        message = f'{path}: launcher must be "slurm" or "local", got {launcher!r}'
        raise MachineProfileError(message)
    output_root = _string_field(table=table, name='output_root', path=path)
    env = _expand_env(table=table, path=path, lenient=lenient_env)
    if launcher == 'local':
        _reject_slurm(table=table, path=path)
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
    try:
        root = expand_variables(text=output_root, env={**os.environ, **env})
    except ValueError as error:
        message = f'{path}: output_root: {error}'
        raise MachineProfileError(message) from error
    tslab_command = _string_tuple(
        table=table, name='tslab_command', path=path, default=('uv', 'run', 'tslab')
    )
    if not tslab_command:
        message = f'{path}: tslab_command must not be empty'
        raise MachineProfileError(message)
    return MachineProfile(
        path=path,
        launcher=launcher,
        output_root=Path(root),
        tslab_command=tslab_command,
        settings=_settings_of(table=table, location='settings', path=path),
        env=env,
        slurm=slurm,
        followup=followup or FollowupSettings(),
        overrides=overrides,
    )


def _reject_slurm(*, table: dict[str, object], path: Path) -> None:
    """A local launcher has no scheduler: SLURM tables would be silently half-applied."""
    if 'slurm' in table:
        message = f'{path}: slurm is not used by launcher "local"; remove the [slurm] table'
        raise MachineProfileError(message)
    entries = table.get('overrides')
    if not isinstance(entries, list):
        return
    for position, entry in enumerate(entries):
        if isinstance(entry, dict) and 'slurm' in entry:
            message = (
                f'{path}: overrides[{position}].slurm is not used by launcher "local"; remove it'
            )
            raise MachineProfileError(message)


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


def _expand_env(*, table: dict[str, object], path: Path, lenient: bool) -> dict[str, str]:
    value = table.get('env', {})
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        message = f'{path}: [env] must hold string values'
        raise MachineProfileError(message)
    expanded: dict[str, str] = {}
    for name, item in value.items():
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            message = (
                f'{path}: [env] name {name!r} is not a shell variable name '
                f'(letters, digits and underscores, not starting with a digit)'
            )
            raise MachineProfileError(message)
        try:
            expanded[name] = expand_variables(text=item, env={**os.environ, **expanded})
        except ValueError as error:
            if not lenient:
                message = f'{path}: [env] value for {name!r}: {error}'
                raise MachineProfileError(message) from error
            expanded[name] = item
    return expanded


def _overrides_table(*, table: dict[str, object], path: Path) -> list[object]:
    value = table.get('overrides', [])
    if not isinstance(value, list):
        message = f'{path}: overrides must be an array of tables'
        raise MachineProfileError(message)
    return cast('list[object]', value)


def _settings_of(*, table: dict[str, object], location: str, path: Path) -> dict[str, SettingValue]:
    """The flattened ``settings`` table of ``table``; every value must be a settings value."""
    value = table.get('settings', {})
    if not isinstance(value, dict):
        message = f'{path}: {location} must be a table, got {type(value).__name__}'
        raise MachineProfileError(message)
    flat = flatten_settings(table=value)
    for key, item in flat.items():
        if not _is_setting_value(value=item):
            message = (
                f'{path}: {location}.{key} holds {type(item).__name__}, which settings cannot '
                f'hold; use strings, numbers, booleans or lists of those'
            )
            raise MachineProfileError(message)
    return flat


def _is_setting_value(*, value: object) -> bool:
    if isinstance(value, str | int | float | bool):
        return True
    return isinstance(value, tuple) and all(_is_setting_value(value=item) for item in value)


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
    if cls is MachineOverride:
        return cast('T', _override_from_table(table=table, location=location, path=path))
    _check_keys(cls=cls, table=table, location=location, path=path)
    instance = _decode(data=table, cls=cls, location=location, path=path)
    for key in table:
        _check_range(key=key, value=getattr(instance, key), location=f'{location}.{key}', path=path)
    return instance


def _decode[T](*, data: dict[str, object], cls: type[T], location: str, path: Path) -> T:
    try:
        return decode_dataclass(data=data, cls=cls, location=location)
    except ConfigFileError as error:
        message = f'{path}: {error}'
        raise MachineProfileError(message) from error


def _check_keys(*, cls: type, table: dict[str, object], location: str, path: Path) -> None:
    known = {item.name for item in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
    unknown = [name for name in table if name not in known]
    if unknown:
        problem = unknown_key_message(location=location, key=unknown[0], valid=known)
        message = f'{path}: {problem}'
        raise MachineProfileError(message)


def _override_from_table(*, table: dict[str, object], location: str, path: Path) -> MachineOverride:
    if 'match' not in table:
        message = f'{path}: {location}.match is required'
        raise MachineProfileError(message)
    _check_keys(cls=MachineOverride, table=table, location=location, path=path)
    slurm = _slurm_overrides(table=table.get('slurm', {}), location=location, path=path)
    settings = _settings_of(table=table, location=f'{location}.settings', path=path)
    if not slurm and not settings:
        message = f'{path}: {location} needs a slurm or a settings table'
        raise MachineProfileError(message)
    return MachineOverride(
        match=_match_value(value=table['match'], location=location, path=path),
        slurm=slurm,
        settings=settings,
    )


def _match_value(*, value: object, location: str, path: Path) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        message = f'{path}: {location}.match must be a non-empty table'
        raise MachineProfileError(message)
    match: dict[str, tuple[str, ...]] = {}
    for key, item in value.items():
        names = [item] if isinstance(item, str) else item
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
        ):
            message = f'{path}: {location}.match.{key} must be a name or a non-empty list of names'
            raise MachineProfileError(message)
        match[key] = tuple(names)
    return match


def _slurm_overrides(*, table: object, location: str, path: Path) -> dict[str, object]:
    """The keys of an override ``slurm`` table, each checked like the base ``[slurm]`` table."""
    if not isinstance(table, dict):
        message = f'{path}: {location}.slurm must be a table'
        raise MachineProfileError(message)
    _check_keys(cls=SlurmSettings, table=table, location=f'{location}.slurm', path=path)
    # why: decode against the full class; the partition of the probe is never read back
    probe = _decode(
        data={'partition': 'probe', **table},
        cls=SlurmSettings,
        location=f'{location}.slurm',
        path=path,
    )
    values: dict[str, object] = {}
    for key in table:
        values[key] = getattr(probe, key)
        _check_range(key=key, value=values[key], location=f'{location}.slurm.{key}', path=path)
    return values


def _check_range(*, key: str, value: object, location: str, path: Path) -> None:
    """Numbers stay in range; text that goes into a job script holds no line break."""
    minimum = _MINIMUMS.get(key)
    if minimum is not None and isinstance(value, int | float) and value < minimum:
        message = f'{path}: {location} must be at least {minimum}, got {value}'
        raise MachineProfileError(message)
    texts = [value] if isinstance(value, str) else list(value) if isinstance(value, tuple) else []
    if any(isinstance(text, str) and re.search(r'[\x00-\x1f\x7f]', text) for text in texts):
        message = f'{path}: {location} must not hold a line break or other control character'
        raise MachineProfileError(message)


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
