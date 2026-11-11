"""The local launch filter that drops downstream tasks whose upstream is not met."""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import TYPE_CHECKING

from tslab.lab.enums import AfterVerdict, JobPhase
from tslab.lab.study.chain import after_verdict
from tslab.lab.study.definition.after import upstream_runs
from tslab.lab.study.launchers.common.types import JobStates
from tslab.lab.study.paths import study_folder
from tslab.lab.study.records import latest_tasks, read_jobs
from tslab.lab.study.reporting.state import collect_states

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tslab.lab.study.context import Context
    from tslab.lab.study.definition.definition import Study, StudyRun
    from tslab.lab.study.plan import JobGroup
    from tslab.lab.study.records import JobRecord

__all__ = ['after_filter']

_LOGGER = logging.getLogger(__name__)


def after_filter(*, context: Context) -> Callable[[JobGroup], JobGroup]:
    """The local hook that drops a task whose upstream dependency is not met.

    The local launch runs its groups in order, so when a group with ``after``
    starts, its upstream groups of the launch have ended. The hook re-reads
    the upstream run states and keeps only the tasks whose condition is met;
    every dropped task is logged one line. A group without ``after`` passes
    unchanged.

    Args:
        context: The context of the launch; the study, machine and folder.

    Returns:
        The hook the launch applies to each group before it is recorded.
    """

    def filter_group(group: JobGroup) -> JobGroup:
        if not group.after:
            return group
        folder = study_folder(machine=context.machine, study=context.study)
        jobs = read_jobs(path=folder / 'submissions.jsonl')
        runs = _upstream_runs(study=context.study, group=group)
        reports = {
            report.run.run_path: report
            for report in collect_states(
                study_directory=context.machine.output_root / context.study.name,
                runs=runs,
                jobs=jobs,
                job_states=_completed_states(runs=runs, jobs=jobs),
            )
        }
        kept = []
        for task in group.tasks:
            verdict = after_verdict(
                study=context.study,
                run=task.resolved.run,
                reports=reports,
                followup=context.machine.followup,
                now=time.time(),
            )
            if verdict is AfterVerdict.MET:
                kept.append(task)
            else:
                _LOGGER.info(
                    'not submitting %s: the upstream dependency is not met',
                    task.resolved.run.run_path,
                )
        if len(kept) == len(group.tasks):
            return group
        return replace(group, tasks=tuple(kept))

    return filter_group


def _upstream_runs(*, study: Study, group: JobGroup) -> list[StudyRun]:
    """The upstream runs of every task of the group, without duplicates."""
    by_path = {
        upstream.run_path: upstream
        for task in group.tasks
        for after in study.grids[task.resolved.run.grid].after
        for upstream in upstream_runs(study=study, run=task.resolved.run, after=after)
    }
    return list(by_path.values())


def _completed_states(*, runs: Sequence[StudyRun], jobs: Sequence[JobRecord]) -> JobStates:
    """Treat the latest task of each run as completed, as it is in this launch."""
    latest = latest_tasks(jobs=jobs)
    phases: dict[str, JobPhase] = {}
    for run in runs:
        record = latest.get(run.run_path)
        if record is not None:
            phases[f'{record[0].job_id}_{record[1].index}'] = JobPhase.COMPLETED
    return JobStates(
        known=True, states={}, phases=phases, reasons={}, flags={}, followup_live_job_ids=()
    )
