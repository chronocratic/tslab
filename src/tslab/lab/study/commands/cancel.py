"""The cancel command: cancel or signal the live jobs of selected runs."""

import logging

from tslab.enums import StudyState
from tslab.study.chain import downstream_of
from tslab.study.context import Context, study_reports, widen_context
from tslab.study.definition import chained_grids
from tslab.study.launchers.common.types import LaunchError
from tslab.study.launchers.tables import backend_for

__all__ = ['cancel']

_LOGGER = logging.getLogger(__name__)


def _with_waiting_downstream(*, context: Context) -> Context:
    """The context widened by the waiting runs that depend on the selected runs."""
    if not chained_grids(study=context.study):
        return context
    everything = study_reports(context=context)
    extra = [
        report
        for report in downstream_of(
            study=context.study,
            run_paths={report.run.run_path for report in context.reports},
            reports=list(everything.values()),
        )
        if report.state is StudyState.WAITING and report.job_id is not None
    ]
    if not extra:
        return context
    _LOGGER.info('also cancelled %d waiting downstream runs', len(extra))
    return widen_context(context=context, reports=extra)


def cancel(*, context: Context, dry_run: bool, signal: str | None = None) -> tuple[int, list[str]]:
    """Cancel or signal the queued and running SLURM jobs of selected runs.

    On a local machine it stops the live local launch with a SIGTERM; the
    selection must cover every live task, because the launch is one unit.

    Args:
        context: The study context containing selected run reports.
        dry_run: Log the job ids without cancelling or signalling them.
        signal: Send this signal to the running jobs instead of cancelling
            them, so the runs save and end on their own; queued runs are
            skipped because a pending job has no step to signal.

    Returns:
        The process exit code and the job ids cancelled, signalled, or
        selected for either.
    """
    if signal is None:
        context = _with_waiting_downstream(context=context)
    try:
        return backend_for(machine=context.machine).cancel(
            context=context, dry_run=dry_run, signal=signal
        )
    except LaunchError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: cancellation failures need no traceback
        return 1, []
