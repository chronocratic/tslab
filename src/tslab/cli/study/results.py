"""The ``tslab study results`` command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import tyro

from tslab.cli.common import resolve_request, use_stderr
from tslab.cli.study.shared import _context_of, _warn_no_runs
from tslab.core.meta import hostname
from tslab.study.commands import results
from tslab.study.commands.results import ResultsFormat

__all__ = ['Results']


@dataclass(frozen=True, kw_only=True)
class Results:
    """Collect the metrics of every run of a study into one CSV file.

    For example:
        tslab study results example --machine local --out results.csv

    Attributes:
        study: The short study name or full module:attribute reference. A short name expands
            against the studies package of tslab.toml.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        out: The CSV file to write.
        json: Write rows as a versioned JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    out: Annotated[Path, tyro.conf.arg(metavar='FILE')]
    json: bool = False

    def run(self) -> int:
        """Write the table and return the exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        context = _context_of(
            study=study, machine=machine, grid=self.grid, only=self.only, read_only=True
        )
        _warn_no_runs(context=context)
        if self.out == Path('-'):
            use_stderr()
        results_format = ResultsFormat.JSON if self.json else ResultsFormat.CSV
        return results(context=context, out=self.out, results_format=results_format)
