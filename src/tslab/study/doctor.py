"""Readiness checks for a study machine and project."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib.util
import logging
import netrc
from pathlib import Path
import shutil
import tempfile
from typing import Literal

from tslab.core.paths import require_path
from tslab.study.loading import load_study
from tslab.study.machine import load_machine, MachineProfile, MachineProfileError
from tslab.study.plan import check_machine

__all__ = ['DoctorCheck', 'run_checks']

_LOGGER = logging.getLogger(__name__)
_DISK_WARNING_BYTES = 10 * 1024**3


@dataclass(frozen=True, kw_only=True)
class DoctorCheck:
    """One result of a doctor check.

    Attributes:
        name: Stable check name.
        status: Whether the check passed, warned, or failed.
        message: Short result detail.
    """

    name: str
    status: Literal['ok', 'warn', 'fail']
    message: str


def run_checks(
    *,
    machine_path: Path,
    study: str | None,
    start: Path,
    environment: Mapping[str, str],
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
) -> list[DoctorCheck]:
    """Check whether a machine and project are ready for a study.

    Args:
        machine_path: The machine profile to check.
        study: Optional study reference to check against the profile.
        start: The working folder to search for a project marker.
        environment: The process environment used by machine and W&B checks.
        free_bytes: Return available disk bytes at a folder.

    Returns:
        Readiness checks in their stable display order.
    """
    require_path(value=machine_path, parameter='machine_path')
    require_path(value=start, parameter='start')
    checks: list[DoctorCheck] = []
    try:
        machine = load_machine(path=machine_path, lenient_env=False, environment=environment)
    except MachineProfileError as error:
        checks.append(DoctorCheck(name='profile', status='fail', message=str(error)))
        machine = None
    else:
        checks.append(DoctorCheck(name='profile', status='ok', message='profile loaded'))

    if machine is not None:
        # why: the launcher package imports DoctorCheck from this module
        from tslab.study.launchers import backend_for  # noqa: PLC0415

        checks.extend(backend_for(machine=machine).doctor_checks())
        output_check, disk_path = _output_root_check(machine=machine)
        checks.append(output_check)
        checks.append(_disk_check(path=disk_path, free_bytes=free_bytes))

    checks.append(_project_check(start=start))
    if study is not None and machine is not None:
        checks.append(_study_check(reference=study, machine=machine))
    if importlib.util.find_spec('wandb') is not None:
        checks.append(_wandb_check(environment=environment))
    return checks


def _output_root_check(*, machine: MachineProfile) -> tuple[DoctorCheck, Path]:
    output_root = machine.output_root
    probe = output_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if not probe.is_dir():
        return (
            DoctorCheck(
                name='output_root',
                status='fail',
                message=f'no existing directory can contain {output_root.as_posix()}',
            ),
            probe,
        )
    try:
        with tempfile.NamedTemporaryFile(dir=probe, delete=True):
            pass
    except OSError as error:
        return (
            DoctorCheck(
                name='output_root',
                status='fail',
                message=f'cannot write to {probe.as_posix()}: {error}',
            ),
            probe,
        )
    if probe == output_root:
        message = f'writable: {output_root.as_posix()}'
    else:
        message = f'writable parent: {probe.as_posix()}'
    return DoctorCheck(name='output_root', status='ok', message=message), probe


def _disk_check(*, path: Path, free_bytes: Callable[[Path], int]) -> DoctorCheck:
    try:
        free = free_bytes(path)
    except OSError as error:
        return DoctorCheck(
            name='disk', status='fail', message=f'cannot check free disk space: {error}'
        )
    status: Literal['ok', 'warn'] = 'warn' if free < _DISK_WARNING_BYTES else 'ok'
    return DoctorCheck(
        name='disk', status=status, message=f'{free / 1024**3:.1f} GB free at {path.as_posix()}'
    )


def _project_check(*, start: Path) -> DoctorCheck:
    folder = start.resolve()
    project = next(
        (
            candidate / 'pyproject.toml'
            for candidate in (folder, *folder.parents)
            if (candidate / 'pyproject.toml').is_file()
        ),
        None,
    )
    if project is None:
        return DoctorCheck(
            name='project',
            status='warn',
            message=f'no pyproject.toml found above {folder.as_posix()}',
        )
    return DoctorCheck(name='project', status='ok', message=f'found {project.as_posix()}')


def _study_check(*, reference: str, machine: MachineProfile) -> DoctorCheck:
    try:
        study = load_study(reference=reference)
        check_machine(study=study, machine=machine)
    except (ImportError, FileNotFoundError, TypeError, ValueError) as error:
        return DoctorCheck(name='study', status='fail', message=str(error))
    return DoctorCheck(name='study', status='ok', message='study fits machine')


def _wandb_check(*, environment: Mapping[str, str]) -> DoctorCheck:
    has_key = bool(environment.get('WANDB_API_KEY'))
    try:
        has_netrc = netrc.netrc().authenticators('api.wandb.ai') is not None
    except (OSError, netrc.NetrcParseError):
        has_netrc = False
    if has_key or has_netrc:
        return DoctorCheck(name='wandb', status='ok', message='W&B credentials found')
    return DoctorCheck(name='wandb', status='warn', message='wandb is installed but not logged in')
