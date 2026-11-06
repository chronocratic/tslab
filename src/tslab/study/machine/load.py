"""Reading a machine profile file into a checked MachineProfile."""

from collections.abc import Mapping
import dataclasses
from pathlib import Path
import tomllib
from typing import cast, Protocol

from tslab.core.text import unknown_key_message
from tslab.enums import ChainedRetry, LauncherKind
from tslab.study.definition import expand_variables
import tslab.study.machine.local as machine_local
from tslab.study.machine.local import LocalSettings
from tslab.study.machine.profile import FollowupSettings, MachineOverride, MachineProfile
import tslab.study.machine.slurm as machine_slurm
from tslab.study.machine.slurm import slurm_overrides, SlurmSettings
from tslab.study.machine.tables import (
    check_keys,
    expand_env,
    from_table,
    MachineProfileError,
    settings_of,
    string_field,
    string_tuple,
)

__all__ = ['load_machine']

_TOP_LEVEL_KEYS = frozenset(
    item.name for item in dataclasses.fields(MachineProfile) if item.name != 'path'
)


class ProfileBackend(Protocol):
    """The machine-specific part of one profile table."""

    @property
    def TABLE(self) -> LauncherKind:  # noqa: N802 # why: the profile module exports TABLE
        """The top-level key that selects this table."""
        ...

    @property
    def FOLLOWUP_KEYS(self) -> frozenset[str]:  # noqa: N802 # why: the profile module exports FOLLOWUP_KEYS
        """The follow-up keys this backend supports."""
        ...

    def settings_from(
        self, *, table: Mapping[str, object], path: Path
    ) -> SlurmSettings | LocalSettings:
        """Read this backend's settings from one profile."""
        ...


PROFILE_BACKENDS: dict[LauncherKind, ProfileBackend] = {
    machine_slurm.TABLE: machine_slurm,
    machine_local.TABLE: machine_local,
}


def load_machine(
    *, path: Path, lenient_env: bool = False, environment: Mapping[str, str]
) -> MachineProfile:
    """The machine profile a TOML file holds, checked key by key.

    Args:
        path: The profile file to read.
        lenient_env: Keep an ``[env]`` value with an unset ``${VAR}`` as it is
            instead of raising, for commands that only read state and never
            launch a job (``output_root`` is always expanded strictly).
        environment: The process environment used to expand profile values.

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
    return _machine_profile(
        path=path, table=table, lenient_env=lenient_env, environment=environment
    )


def _machine_profile(
    *, path: Path, table: dict[str, object], lenient_env: bool, environment: Mapping[str, str]
) -> MachineProfile:
    unknown = [name for name in table if name not in _TOP_LEVEL_KEYS]
    if unknown:
        problem = unknown_key_message(location='', key=unknown[0], valid=_TOP_LEVEL_KEYS)
        message = f'{path}: {problem}'
        raise MachineProfileError(message)
    launcher = string_field(table=table, name='launcher', path=path)
    try:
        kind = LauncherKind(launcher)
    except ValueError as error:
        message = f'{path}: launcher must be "slurm" or "local", got {launcher!r}'
        raise MachineProfileError(message) from error
    backend = PROFILE_BACKENDS[kind]
    output_root = string_field(table=table, name='output_root', path=path)
    env = expand_env(table=table, path=path, lenient=lenient_env, environment=environment)
    _reject_other_backend_tables(backend=backend, table=table, path=path)
    _reject_unsupported_followups(backend=backend, table=table, path=path)
    _reject_renamed_auto_retry(table=table, path=path)
    backend_settings = backend.settings_from(table=table, path=path)
    slurm = backend_settings if isinstance(backend_settings, SlurmSettings) else None
    local = backend_settings if isinstance(backend_settings, LocalSettings) else None
    followup = from_table(
        cls=FollowupSettings,
        table=table.get('followup'),
        location='followup',
        path=path,
        required=False,
    )
    if (
        backend.TABLE is LauncherKind.LOCAL
        and followup is not None
        and followup.chained_retry is ChainedRetry.REQUEUE
    ):
        message = (
            f'{path}: followup.chained_retry = "requeue" is SLURM only; use "manual" or "resubmit"'
        )
        raise MachineProfileError(message)
    overrides = tuple(
        _override_from_table(table=entry, location=f'overrides[{position}]', path=path)
        for position, entry in enumerate(_overrides_table(table=table, path=path))
    )
    try:
        root = expand_variables(text=output_root, env={**environment, **env})
    except ValueError as error:
        message = f'{path}: output_root: {error}'
        raise MachineProfileError(message) from error
    tslab_command = string_tuple(
        table=table, name='tslab_command', path=path, default=('uv', 'run', 'tslab')
    )
    if not tslab_command:
        message = f'{path}: tslab_command must not be empty'
        raise MachineProfileError(message)
    return MachineProfile(
        path=path,
        launcher=backend.TABLE,
        output_root=Path(root),
        tslab_command=tslab_command,
        settings=settings_of(table=table, location='settings', path=path),
        env=env,
        slurm=slurm,
        local=local,
        followup=followup or FollowupSettings(),
        overrides=overrides,
    )


def _reject_other_backend_tables(
    *, backend: ProfileBackend, table: dict[str, object], path: Path
) -> None:
    for other in PROFILE_BACKENDS.values():
        if other is backend:
            continue
        _reject_other_backend_table(backend=backend, other=other, table=table, path=path)


def _reject_other_backend_table(
    *, backend: ProfileBackend, other: ProfileBackend, table: dict[str, object], path: Path
) -> None:
    if other.TABLE in table:
        message = (
            f'{path}: {other.TABLE} is not used by launcher "{backend.TABLE}"; '
            f'remove the [{other.TABLE}] table'
        )
        raise MachineProfileError(message)
    entries = table.get('overrides')
    if not isinstance(entries, list):
        return
    position = next(
        (
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, dict) and other.TABLE in entry
        ),
        None,
    )
    if position is None:
        return
    message = (
        f'{path}: overrides[{position}].{other.TABLE} is not used by launcher '
        f'"{backend.TABLE}"; remove it'
    )
    raise MachineProfileError(message)


def _reject_unsupported_followups(
    *, backend: ProfileBackend, table: dict[str, object], path: Path
) -> None:
    followup = table.get('followup')
    if not isinstance(followup, dict):
        return
    for key in followup:
        owner = next(
            (
                other
                for other in PROFILE_BACKENDS.values()
                if other is not backend
                and key in other.FOLLOWUP_KEYS
                and key not in backend.FOLLOWUP_KEYS
            ),
            None,
        )
        if owner is None:
            continue
        message = f'{path}: followup.{key} is {owner.TABLE.upper()} only; remove it'
        raise MachineProfileError(message)


def _reject_renamed_auto_retry(*, table: Mapping[str, object], path: Path) -> None:
    """Reject the removed profile key with its replacement."""
    followup = table.get('followup')
    if not isinstance(followup, dict) or 'auto_retry' not in followup:
        return
    message = f'{path}: followup.auto_retry` was renamed: use `retry = "resubmit"` or `"manual"`'
    raise MachineProfileError(message)


def _override_from_table(*, table: object, location: str, path: Path) -> MachineOverride:
    if table is None:
        message = f'{path}: the {location} table is required'
        raise MachineProfileError(message)
    if not isinstance(table, dict):
        message = f'{path}: {location} must be a table, got {type(table).__name__}'
        raise MachineProfileError(message)
    if 'match' not in table:
        message = f'{path}: {location}.match is required'
        raise MachineProfileError(message)
    check_keys(cls=MachineOverride, table=table, location=location, path=path)
    slurm = slurm_overrides(table=table.get('slurm', {}), location=location, path=path)
    settings = settings_of(table=table, location=f'{location}.settings', path=path)
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


def _overrides_table(*, table: dict[str, object], path: Path) -> list[object]:
    value = table.get('overrides', [])
    if not isinstance(value, list):
        message = f'{path}: overrides must be an array of tables'
        raise MachineProfileError(message)
    return cast('list[object]', value)
