"""The ``tslab doctor`` readiness command."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated

import tyro

from tslab.cli.output import print_json
from tslab.cli.usage import UsageError
from tslab.core.text import log_table
from tslab.study.doctor import run_checks
from tslab.study.machine import resolve_machine

__all__ = ['Doctor']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Doctor:
    """Check whether a machine and optional study are ready.

    For example:
        tslab doctor --machine local
        tslab doctor --machine cluster my_paper.studies:example

    Attributes:
        machine: A machine name or profile path.
        study: Optional study reference, module:attribute.
        json: Print one doctor JSON document.
    """

    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    study: Annotated[str | None, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional] = None
    json: bool = False

    def run(self) -> int:
        """Run readiness checks and return the command exit code."""
        try:
            location = resolve_machine(machine=self.machine, start=Path.cwd())
        except (ValueError, TypeError, FileNotFoundError) as error:
            raise UsageError(str(error)) from error
        checks = run_checks(machine_path=location.path, study=self.study, start=Path.cwd())
        code = int(any(check.status == 'fail' for check in checks))
        if self.json:
            print_json(
                document={
                    'schema': 'tslab.doctor/1',
                    'machine': location.path.resolve().as_posix(),
                    'checks': [
                        {'name': check.name, 'status': check.status, 'message': check.message}
                        for check in checks
                    ],
                    'exit_code': code,
                }
            )
        else:
            log_table(
                logger=_LOGGER,
                header=('check', 'status', 'message'),
                rows=[(check.name, check.status, check.message) for check in checks],
            )
            _LOGGER.info('ready' if code == 0 else 'fix the failed checks first')
        return code
