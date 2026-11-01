"""Reading a machine profile file into a checked MachineProfile."""

import dataclasses
import os
from pathlib import Path
import tomllib
from typing import cast, TYPE_CHECKING

from tslab.core.text import unknown_key_message
from tslab.study.expansion import expand_variables
from tslab.study.machine.local import (
    check_local,
    local_gpus,
    LocalSettings,
    reject_local_followups,
    reject_slurm,
)
from tslab.study.machine.profile import FollowupSettings, MachineOverride, MachineProfile
from tslab.study.machine.slurm import reject_local, sbatch_extra, slurm_overrides, SlurmSettings
from tslab.study.machine.tables import (
    check_keys,
    expand_env,
    from_table,
    MachineProfileError,
    settings_of,
    string_field,
    string_tuple,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['load_machine']

_TOP_LEVEL_KEYS = frozenset(
    item.name for item in dataclasses.fields(MachineProfile) if item.name != 'path'
)


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
    launcher = string_field(table=table, name='launcher', path=path)
    if launcher not in {'slurm', 'local'}:
        message = f'{path}: launcher must be "slurm" or "local", got {launcher!r}'
        raise MachineProfileError(message)
    output_root = string_field(table=table, name='output_root', path=path)
    env = expand_env(table=table, path=path, lenient=lenient_env)
    if launcher == 'local':
        reject_slurm(table=table, path=path)
        reject_local_followups(table=table, path=path)
    else:
        reject_local(table=table, path=path)
    slurm_table = table.get('slurm')
    if isinstance(slurm_table, dict) and 'sbatch_extra' in slurm_table:
        slurm_table = {
            **slurm_table,
            'sbatch_extra': [
                list(pair)
                for pair in sbatch_extra(
                    value=slurm_table['sbatch_extra'], location='slurm.sbatch_extra', path=path
                )
            ],
        }
    slurm = from_table(
        cls=SlurmSettings,
        table=slurm_table,
        location='slurm',
        path=path,
        required=launcher == 'slurm',
    )
    local_table = table.get('local')
    if isinstance(local_table, dict) and 'gpus' in local_table:
        local_table = {
            **local_table,
            'gpus': local_gpus(value=local_table['gpus'], location='local.gpus', path=path),
        }
    local = from_table(
        cls=LocalSettings, table=local_table, location='local', path=path, required=False
    )
    if launcher == 'local':
        local = local or LocalSettings()
        # why: from_table raised when the local table is not a dict
        check_local(
            local=local, table=cast('Mapping[str, object]', table.get('local') or {}), path=path
        )
    followup = from_table(
        cls=FollowupSettings,
        table=table.get('followup'),
        location='followup',
        path=path,
        required=False,
    )
    overrides = tuple(
        _override_from_table(table=entry, location=f'overrides[{position}]', path=path)
        for position, entry in enumerate(_overrides_table(table=table, path=path))
    )
    try:
        root = expand_variables(text=output_root, env={**os.environ, **env})
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
        launcher=launcher,
        output_root=Path(root),
        tslab_command=tslab_command,
        settings=settings_of(table=table, location='settings', path=path),
        env=env,
        slurm=slurm,
        local=(local or LocalSettings()) if launcher == 'local' else None,
        followup=followup or FollowupSettings(),
        overrides=overrides,
    )


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
