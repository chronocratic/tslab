"""The ``tslab doctor`` readiness command."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Annotated, cast

import tyro

from tslab.cli.common import print_json, resolve_request, UsageError
from tslab.core.enums import CheckStatus
from tslab.core.meta import hostname
from tslab.core.text import log_table
from tslab.study.doctor import run_checks
from tslab.study.machine import resolve_machine
from tslab.study.project import ProjectSettings

__all__ = ['Doctor']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Doctor:
    """Check whether a machine and optional study are ready.

    For example:
        tslab doctor --machine local
        tslab doctor --machine cluster my_paper.studies:example

    Attributes:
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
        study: Optional study reference, module:attribute. A short name is
            expanded against the studies module of tslab.toml.
        json: Print one doctor JSON document.
    """

    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    study: Annotated[str | None, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional] = None
    json: bool = False

    def run(self) -> int:
        """Run readiness checks and return the command exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        machine = cast('str', request.machine)
        try:
            location = resolve_machine(machine=machine, start=Path.cwd(), environment=os.environ)
        except (ValueError, TypeError, FileNotFoundError) as error:
            raise UsageError(str(error)) from error
        checks = run_checks(
            machine_path=location.path,
            study=request.study,
            start=Path.cwd(),
            environment=os.environ,
        )
        code = int(any(check.status == CheckStatus.FAIL for check in checks))
        if self.json:
            print_json(
                document={
                    'schema': 'tslab.doctor/1',
                    'machine': location.path.resolve().as_posix(),
                    'project_settings': _project_settings_document(
                        settings=request.settings, machine=self.machine
                    ),
                    'checks': [
                        {'name': check.name, 'status': check.status, 'message': check.message}
                        for check in checks
                    ],
                    'exit_code': code,
                }
            )
        else:
            _log_project_settings(settings=request.settings, machine=self.machine)
            log_table(
                logger=_LOGGER,
                header=('check', 'status', 'message'),
                rows=[(check.name, check.status, check.message) for check in checks],
            )
            _LOGGER.info('ready' if code == 0 else 'fix the failed checks first')
        return code


def _source_of(*, keys: frozenset[str], key: str) -> str:
    """The source of one key: file when the key is in the file, else default."""
    if key in keys:
        return 'file'
    return 'default'


def _machine_entry(
    *, settings: ProjectSettings, keys: frozenset[str], machine: str | None
) -> tuple[object, str]:
    """The machine_by_host value and source of the JSON document; the flag wins."""
    if machine is not None:
        return machine, 'flag'
    pairs = [[pattern, name] for pattern, name in settings.machine_by_host]
    return pairs, _source_of(keys=keys, key='machine_by_host')


def _machine_row(
    *, settings: ProjectSettings, keys: frozenset[str], machine: str | None
) -> tuple[str, str]:
    """The machine_by_host value and source of the text table; the flag wins."""
    if machine is not None:
        return machine, 'flag'
    pairs = [f'{pattern} = {name}' for pattern, name in settings.machine_by_host]
    return ', '.join(pairs) if pairs else 'none', _source_of(keys=keys, key='machine_by_host')


def _log_project_settings(*, settings: ProjectSettings, machine: str | None) -> None:
    """The project settings as one log line and a key, value, source table."""
    keys = settings.file_keys
    path = settings.path
    _LOGGER.info('project settings: %s', path.as_posix() if path is not None else 'none')
    studies = settings.studies
    runs = settings.runs
    host_value, host_source = _machine_row(settings=settings, keys=keys, machine=machine)
    log_table(
        logger=_LOGGER,
        header=('key', 'value', 'source'),
        rows=[
            (
                'studies',
                studies if studies is not None else 'none',
                _source_of(keys=keys, key='studies'),
            ),
            ('machine_by_host', host_value, host_source),
            (
                'runs',
                runs.as_posix() if runs is not None else 'none',
                _source_of(keys=keys, key='runs'),
            ),
        ],
    )


def _project_settings_document(
    *, settings: ProjectSettings, machine: str | None
) -> dict[str, object]:
    """The project_settings entry of the doctor JSON document."""
    keys = settings.file_keys
    host_value, host_source = _machine_entry(settings=settings, keys=keys, machine=machine)
    runs = settings.runs
    studies = settings.studies
    return {
        'path': settings.path.as_posix() if settings.path is not None else None,
        'values': {
            'studies': {'value': studies, 'source': _source_of(keys=keys, key='studies')},
            'machine_by_host': {'value': host_value, 'source': host_source},
            'runs': {
                'value': runs.as_posix() if runs is not None else None,
                'source': _source_of(keys=keys, key='runs'),
            },
        },
    }
