"""Condition verdicts for dependencies between study grids."""

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso
from tslab.lab.enums import (
    AfterCondition,
    AfterVerdict,
    ChainedRetry,
    InstanceState,
    JobPhase,
    RunState,
    Satisfy,
    StudyState,
)
from tslab.lab.study import After, Grid, Study
from tslab.lab.study.chain import _VERDICT_BY_CONDITION, after_verdict, retry_left, upstream_verdict
from tslab.lab.study.definition import expand_runs, StudyRun
from tslab.lab.study.machine.profile import FollowupSettings
from tslab.lab.study.records import ProgressSnapshot, TaskRecord
from tslab.lab.study.reporting.state import FolderFacts, RunReport


@dataclass(frozen=True, kw_only=True)
class _ChainConfig:
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _study(
    *, satisfy: Satisfy = Satisfy.ALL, condition: AfterCondition = AfterCondition.OK
) -> Study:
    upstream = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}},
        run_path='{grid}/{dataset}',
    )
    downstream = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainConfig,
        axes={'dataset': {'all': {}}},
        after=(After(grid='pretrain', match=(), satisfy=satisfy, condition=condition),),
        run_path='{grid}/{dataset}',
    )
    return Study(name='chain', grids={'pretrain': upstream, 'probe': downstream})


def _report(
    *, run: StudyRun, state: StudyState, folder: FolderFacts | None = None, attempt: int = 1
) -> RunReport:
    return RunReport(
        run=run,
        state=state,
        reason=None,
        job_id=None,
        job_state=None,
        job_phase=JobPhase.UNKNOWN,
        attempt=attempt,
        task=None,
        folder=folder or FolderFacts(snapshot=None, read_error=None, lock_live=False),
    )


def test_upstream_verdict_table() -> None:
    cases = (
        (AfterCondition.OK, StudyState.DONE, False, AfterVerdict.MET),
        (AfterCondition.OK, StudyState.FAILED, False, AfterVerdict.NEVER),
        (AfterCondition.OK, StudyState.INTERRUPTED, True, AfterVerdict.OPEN),
        (AfterCondition.OK, StudyState.INTERRUPTED, False, AfterVerdict.NEVER),
        (AfterCondition.ANY, StudyState.FAILED, False, AfterVerdict.MET),
        (AfterCondition.FAILED, StudyState.DONE, False, AfterVerdict.NEVER),
        (AfterCondition.FAILED, StudyState.FAILED, True, AfterVerdict.MET),
        (AfterCondition.FAILED, StudyState.INTERRUPTED, True, AfterVerdict.OPEN),
        (AfterCondition.FAILED, StudyState.INTERRUPTED, False, AfterVerdict.MET),
        (AfterCondition.FAILED, StudyState.DEPENDENCY_FAILED, False, AfterVerdict.MET),
        (AfterCondition.STARTED, StudyState.DEPENDENCY_FAILED, False, AfterVerdict.NEVER),
        (AfterCondition.OK, StudyState.QUEUED, False, AfterVerdict.OPEN),
    )

    assert [
        upstream_verdict(condition=condition, state=state, retry_left=retry_left)
        for condition, state, retry_left, _ in cases
    ] == [expected for _, _, _, expected in cases]


def test_every_after_condition_has_a_verdict_strategy() -> None:
    assert set(_VERDICT_BY_CONDITION) == set(AfterCondition)


def test_retry_left_obeys_the_attempt_limit() -> None:
    study = _study()
    run = next(run for run in expand_runs(study=study) if run.grid == 'pretrain')
    followup = FollowupSettings(chained_retry=ChainedRetry.RESUBMIT, max_attempts=3)

    assert retry_left(
        report=_report(run=run, state=StudyState.INTERRUPTED, attempt=2), followup=followup
    )
    assert not retry_left(
        report=_report(run=run, state=StudyState.INTERRUPTED, attempt=3), followup=followup
    )


def test_started_condition_uses_recorded_start_time() -> None:
    assert (
        upstream_verdict(
            condition=AfterCondition.STARTED,
            state=StudyState.RUNNING,
            retry_left=False,
            delay_minutes=30,
            started_at=100.0,
            now=1_900.0,
        )
        is AfterVerdict.MET
    )
    assert (
        upstream_verdict(
            condition=AfterCondition.STARTED,
            state=StudyState.RUNNING,
            retry_left=False,
            delay_minutes=30,
            now=1_900.0,
        )
        is AfterVerdict.OPEN
    )


def test_started_condition_with_zero_delay_needs_no_recorded_start_time() -> None:
    assert (
        upstream_verdict(
            condition=AfterCondition.STARTED, state=StudyState.RUNNING, retry_left=False
        )
        is AfterVerdict.MET
    )
    assert (
        upstream_verdict(condition=AfterCondition.STARTED, state=StudyState.DONE, retry_left=False)
        is AfterVerdict.MET
    )


def test_satisfy_any_needs_one_met() -> None:
    study = _study(satisfy=Satisfy.ANY)
    runs = expand_runs(study=study)
    probe = next(run for run in runs if run.grid == 'probe')
    upstream = {
        run.run_path: _report(
            run=run,
            state=(
                StudyState.DONE
                if run.run_path == PurePosixPath('pretrain/a')
                else StudyState.FAILED
            ),
        )
        for run in runs
        if run.grid == 'pretrain'
    }

    assert after_verdict(study=study, run=probe, reports=upstream) is AfterVerdict.MET
    assert (
        after_verdict(
            study=study,
            run=probe,
            reports={
                path: _report(run=report.run, state=StudyState.FAILED)
                for path, report in upstream.items()
            },
        )
        is AfterVerdict.NEVER
    )


def test_satisfy_all_is_never_when_one_upstream_fails() -> None:
    study = _study()
    runs = expand_runs(study=study)
    probe = next(run for run in runs if run.grid == 'probe')
    upstream = {
        run.run_path: _report(
            run=run,
            state=(
                StudyState.DONE
                if run.run_path == PurePosixPath('pretrain/a')
                else StudyState.FAILED
            ),
        )
        for run in runs
        if run.grid == 'pretrain'
    }

    assert after_verdict(study=study, run=probe, reports=upstream) is AfterVerdict.NEVER


def test_failed_instance_in_finished_upstream_blocks_ok_downstream() -> None:
    study = _study()
    runs = expand_runs(study=study)
    probe = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))
    finished_with_failure = _report(
        run=upstream,
        state=StudyState.FAILED,
        folder=FolderFacts(
            snapshot=RunStatusSnapshot(
                run_state=RunState.FINISHED,
                updated_at=utc_now_iso(),
                instances={'train': InstanceRecord(state=InstanceState.FAILED)},
            ),
            read_error=None,
            lock_live=False,
        ),
    )
    upstream_reports = {
        upstream.run_path: finished_with_failure,
        PurePosixPath('pretrain/b'): _report(
            run=next(run for run in runs if run.run_path == PurePosixPath('pretrain/b')),
            state=StudyState.DONE,
        ),
    }

    assert after_verdict(study=study, run=probe, reports=upstream_reports) is AfterVerdict.NEVER


def test_uncounted_attempts_leave_retries_open() -> None:
    study = _study()
    runs = expand_runs(study=study)
    probe = next(run for run in runs if run.grid == 'probe')
    followup = FollowupSettings(chained_retry=ChainedRetry.REQUEUE, max_attempts=3)

    def interrupted(*, run: StudyRun) -> RunReport:
        # why: attempt 3 after two requeues by the scheduler counts as one attempt
        task = TaskRecord(
            index=0,
            grid=run.grid,
            run_path=run.run_path,
            command=(),
            extra_args=(),
            resources={},
            attempt=3,
            progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
            uncounted_attempts=2,
        )
        return replace(_report(run=run, state=StudyState.INTERRUPTED, attempt=3), task=task)

    upstream = {run.run_path: interrupted(run=run) for run in runs if run.grid == 'pretrain'}

    assert (
        after_verdict(study=study, run=probe, reports=upstream, followup=followup)
        is AfterVerdict.OPEN
    )


def test_failed_condition_meets_interrupted_upstream_at_attempt_limit() -> None:
    study = _study(condition=AfterCondition.FAILED)
    run = next(run for run in expand_runs(study=study) if run.grid == 'probe')
    upstream_runs = [run for run in expand_runs(study=study) if run.grid == 'pretrain']
    upstream = {
        item.run_path: _report(run=item, state=StudyState.INTERRUPTED, attempt=3)
        for item in upstream_runs
    }

    verdict = after_verdict(
        study=study,
        run=run,
        reports=upstream,
        followup=FollowupSettings(chained_retry=ChainedRetry.RESUBMIT, max_attempts=3),
    )

    assert verdict is AfterVerdict.MET
