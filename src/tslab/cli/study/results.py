"""The ``tslab study results`` command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from tslab.cli.common import use_stderr
from tslab.cli.study import _context_of, _warn_no_runs
from tslab.study.commands import results

__all__ = ['Results']


@dataclass(frozen=True, kw_only=True)
class Results:
    """Collect the metrics of every run of a study into one CSV file.

    For example:
        tslab study results my_paper.studies:example --machine local --out results.csv

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        out: The CSV file to write.
        json: Write rows as a versioned JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    out: Annotated[Path, tyro.conf.arg(metavar='FILE')]
    json: bool = False

    def run(self) -> int:
        """Write the table and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        _warn_no_runs(context=context)
        if self.out == Path('-'):
            use_stderr()
        return results(context=context, out=self.out, as_json=self.json)
