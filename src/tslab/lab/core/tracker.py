"""The experiment-tracker interface and the local tracker every run has.

``Run`` talks to trackers only through ``Tracker``. The interface lives in
``core`` because the engine depends on it; users import it from ``tslab``.
``LocalTracker`` needs no service and no extra package, so ``Run`` can always
add it.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import csv
from enum import StrEnum
import io
from pathlib import Path
import re
import shutil
from typing import override

import yaml

from tslab.lab.core.files import atomic_write_text
from tslab.lab.core.paths import TRACKERS_DIRECTORY
from tslab.lab.enums import RunState

__all__ = ['STEP_COLUMN', 'LocalTracker', 'RunStatus', 'Tracker']

_VERSION_PATTERN = re.compile(r'version_(\d+)')
STEP_COLUMN = 'step'


class _ParamsDumper(yaml.SafeDumper):
    """A safe dumper that writes any value it cannot represent as its text."""


# A trainer passes the raw init arguments of a model, which can hold enums, paths
# or classes; safe_dump refuses those, and the whole hparams.yaml would be lost.
_ParamsDumper.add_multi_representer(object, lambda dumper, value: dumper.represent_str(str(value)))


class RunStatus(StrEnum):
    """Final status reported to a tracker when the run ends.

    It has no ``RUNNING`` member: a tracker is only told once the run is over.
    ``from_run_state`` is the explicit mapping from ``RunState``.
    """

    FINISHED = 'finished'
    FAILED = 'failed'
    CRASHED = 'crashed'

    @classmethod
    def from_run_state(cls, *, run_state: RunState) -> 'RunStatus':
        """Return the status that reports a run that ended in ``run_state``.

        Args:
            run_state: A final state of the run.

        Raises:
            ValueError: If ``run_state`` is ``RUNNING``, which is not final.
        """
        mapping = {
            RunState.FINISHED: cls.FINISHED,
            RunState.FAILED: cls.FAILED,
            RunState.CRASHED: cls.CRASHED,
        }
        if run_state not in mapping:
            message = f'a run in state {run_state.value!r} has no final status'
            raise ValueError(message)
        return mapping[run_state]


class Tracker(ABC):
    """Sends parameters, metrics and files of one run to an experiment tracker.

    ``Run`` calls ``start`` once on enter (the run folder is only known then),
    the ``log_*`` methods while the run is active, and ``finish`` once on exit.
    ``Run`` catches and logs every exception a tracker raises, so a tracker
    outage never stops the experiment.
    """

    @abstractmethod
    def start(self, *, run_directory: Path, run_name: str) -> None:
        """Open the tracker run.

        Args:
            run_directory: The run folder; trackers write under ``run_directory / "trackers"``.
            run_name: Relative path of the run in POSIX form, for display.
        """

    @abstractmethod
    def log_params(self, *, params: Mapping[str, object]) -> None:
        """Record flat configuration parameters (keys joined with ``/``)."""

    @abstractmethod
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record scalar metrics, optionally at a step."""

    @abstractmethod
    def log_file(self, *, path: Path) -> None:
        """Upload or copy a file to the tracker run."""

    @abstractmethod
    def finish(self, *, status: RunStatus) -> None:
        """Close the tracker run with its final status."""


class LocalTracker(Tracker):
    """Writes ``trackers/csv/version_<n>/metrics.csv`` and ``hparams.yaml``.

    This is the record that survives when every remote tracker is down, so
    ``Run`` always adds it. It uses only the standard library and PyYAML, which
    keeps ``tslab.lab.core`` free of heavy dependencies.

    Each execution of the run gets the next ``version_<n>`` folder, so a resumed
    run never overwrites the metrics of the interrupted one. Columns are
    ``step`` followed by the sorted metric names; the file is rewritten when a
    new metric name appears and appended to otherwise.
    """

    def __init__(self) -> None:
        """Create an unstarted tracker."""
        self._directory: Path | None = None
        self._files_directory: Path | None = None
        self._rows: list[dict[str, float | int | None]] = []
        self._columns: list[str] = []

    @override
    def start(self, *, run_directory: Path, run_name: str) -> None:
        root = run_directory / TRACKERS_DIRECTORY / 'csv'
        taken = [
            int(match.group(1))
            for item in (root.iterdir() if root.is_dir() else ())
            if (match := _VERSION_PATTERN.fullmatch(item.name)) is not None
        ]
        self._directory = root / f'version_{max(taken, default=-1) + 1}'
        self._directory.mkdir(parents=True)
        self._files_directory = run_directory / TRACKERS_DIRECTORY / 'files'

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        if self._directory is not None:
            text = yaml.dump(dict(params), Dumper=_ParamsDumper, sort_keys=True, allow_unicode=True)
            atomic_write_text(path=self._directory / 'hparams.yaml', text=text)

    @override
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self._directory is None:
            return
        if STEP_COLUMN in metrics:
            message = (
                f'a metric cannot be named {STEP_COLUMN!r}: it is the step column of metrics.csv'
            )
            raise ValueError(message)
        row: dict[str, float | int | None] = {STEP_COLUMN: step, **metrics}
        self._rows.append(row)
        columns = [STEP_COLUMN, *sorted(metrics)]
        known = set(self._columns)
        path = self._directory / 'metrics.csv'
        if known.issuperset(columns):
            with path.open(mode='a', encoding='utf-8', newline='') as handle:
                csv.DictWriter(handle, fieldnames=self._columns).writerow(row)
            return
        self._columns = [STEP_COLUMN, *sorted((known | set(columns)) - {STEP_COLUMN})]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=self._columns)
        writer.writeheader()
        writer.writerows(self._rows)
        atomic_write_text(path=path, text=buffer.getvalue())

    @override
    def log_file(self, *, path: Path) -> None:
        """Copy the file to ``trackers/files/``."""
        if self._files_directory is not None:
            self._files_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self._files_directory / path.name)

    @override
    def finish(self, *, status: RunStatus) -> None:
        """Nothing to close; every write is already on disk."""
