"""The ``tslab study list`` command."""

from dataclasses import dataclass
import logging
from pathlib import Path

from tslab.lab.cli.common.errors import UsageError
from tslab.lab.cli.common.output import print_json
from tslab.lab.cli.common.project import resolve_request
from tslab.lab.cli.study.shared import _usage_errors
from tslab.lab.core.meta.run import hostname
from tslab.lab.study.definition.loading import load_study
from tslab.lab.study.project import expand_study, find_studies

__all__ = ['List']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class List:
    """List public studies found under the configured studies package.

    For example:
        tslab study list

    Attributes:
        json: Print a JSON document with schema tslab.study-list.
    """

    json: bool = False

    def run(self) -> int:
        """Load each discoverable study and print its short and declared names."""
        request = resolve_request(
            study=None, machine=None, start=Path.cwd(), host=hostname(), needs_machine=False
        )
        try:
            short_names = find_studies(settings=request.settings)
        except (ModuleNotFoundError, ValueError) as error:
            package = request.settings.studies
            message = f'cannot import studies package {package!r}: {error}'
            raise UsageError(message) from error
        studies: list[dict[str, str]] = []
        for short_name in short_names:
            reference = expand_study(reference=short_name, settings=request.settings)
            with _usage_errors(study=reference):
                study = load_study(reference=reference)
            studies.append({'short_name': short_name, 'name': study.name})
        if self.json:
            # When API is stable: version this id (tslab.study-list/1); bump on a breaking change.
            print_json(document={'schema': 'tslab.study-list', 'studies': studies})
            return 0
        for study in studies:
            _LOGGER.info('%s  %s', study['short_name'], study['name'])
        return 0
