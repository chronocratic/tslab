"""The study command arguments as full values: the expanded study and the machine.

The typed study and machine are turned into full values here, before
anything is loaded or recorded. A short study name expands against the
``studies`` package of the nearest ``tslab.toml``; a missing machine is
picked from its ``[machine_by_host]`` table for this host. Every later
step sees only the full reference and the machine name.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

from tslab.cli.common.errors import UsageError
from tslab.study.project import (
    expand_study,
    load_project_settings,
    machine_for_host,
    PROJECT_FILE,
    ProjectSettings,
)

__all__ = ['Request', 'resolve_request']


@dataclass(frozen=True, kw_only=True)
class Request:
    """The study and machine of one study command, fully expanded.

    Attributes:
        study: The full ``module:attribute`` reference; ``None`` only
            when the caller passed ``None``.
        machine: The machine name from the flag, or the name picked for
            this host; ``None`` only when the caller passed
            ``needs_machine=False``.
        settings: The project settings the study was expanded against.
    """

    study: str | None
    machine: str | None
    settings: ProjectSettings


def resolve_request(
    *, study: str | None, machine: str | None, start: Path, host: str, needs_machine: bool = True
) -> Request:
    """The full study reference and machine of one study command.

    Args:
        study: The study as typed, short or full.
        machine: The machine of the ``--machine`` flag; ``None`` when the
            flag was not passed.
        start: The folder to look for ``tslab.toml`` from, usually the
            working folder.
        host: The name of the host the command runs on.
        needs_machine: Whether the command needs a machine; a command
            that only loads the study passes ``False``.

    Returns:
        The expanded request.

    Raises:
        UsageError: If the nearest ``tslab.toml`` is broken, or if the
            command needs a machine and neither the flag nor a matching
            pattern gives one.
    """
    try:
        settings = load_project_settings(start=start)
    except ValueError as error:
        raise UsageError(str(error)) from error
    reference = expand_study(reference=study, settings=settings) if study is not None else None
    if machine is not None:
        return Request(study=reference, machine=machine, settings=settings)
    picked = machine_for_host(settings=settings, host=host)
    if picked is not None:
        pattern, name = picked
        line = f'machine {name} from {settings.path} (host {host} matches {pattern!r})\n'
        sys.stderr.write(line)
        return Request(study=reference, machine=name, settings=settings)
    if not needs_machine:
        return Request(study=reference, machine=None, settings=settings)
    raise UsageError(_no_machine_message(settings=settings, host=host))


def _no_machine_message(*, settings: ProjectSettings, host: str) -> str:
    """The usage error of a command that needs a machine and got none."""
    if settings.path is None:
        return (
            f'no machine for host {host!r}: no {PROJECT_FILE} in the working folder or a '
            'folder above it; pass --machine'
        )
    if 'machine_by_host' not in settings.file_keys:
        return (
            f'no machine for host {host!r}: no [machine_by_host] in {settings.path}; pass --machine'
        )
    if not settings.machine_by_host:
        return (
            f'no machine for host {host!r}: [machine_by_host] in {settings.path} has no '
            'pattern; pass --machine'
        )
    patterns = ', '.join(repr(pattern) for pattern, _ in settings.machine_by_host)
    return (
        f'no machine for host {host!r}: no [machine_by_host] pattern in {settings.path} '
        f'matches (patterns: {patterns}); pass --machine'
    )
