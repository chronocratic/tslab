"""Run state changes derived from chained dependencies."""

from dataclasses import dataclass
from pathlib import Path

from tslab.enums import AfterCondition, JobFlag, JobPhase, StudyState
from tslab.study import After, Grid, Study
from tslab.study.definition import expand_runs, StudyRun
from tslab.study.machine.profile import FollowupSettings
from tslab.study.reporting.chain import apply_chain_states
from tslab.study.reporting.state import FolderFacts, RunReport


@dataclass(frozen=True, kw_only=True)
class _ChainConfig:
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _study(*, condition: AfterCondition = AfterCondition.OK) -> Study:
    upstream = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}}},
        run_path='{grid}/{dataset}',
    )
    downstream = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}}},
        after=(After(grid='pretrain', condition=condition),),
        run_path='{grid}/{dataset}',
    )
    return Study(name='chain', grids={'pretrain': upstream, 'probe': downstream})


def _report(
    *,
    run: StudyRun,
    state: StudyState,
    job_state: str | None = None,
    job_flags: frozenset[JobFlag] = frozenset(),
    folder: FolderFacts | None = None,
) -> RunReport:
    return RunReport(
        run=run,
        state=state,
        reason=None,
        job_id=None,
        job_state=job_state,
        job_phase=JobPhase.UNKNOWN,
        attempt=1,
        task=None,
        folder=folder or FolderFacts(snapshot=None, read_error=None, lock_live=False),
        job_flags=job_flags,
    )


def _runs() -> tuple[Study, StudyRun, StudyRun]:
    study = _study()
    runs = expand_runs(study=study)
    upstream = next(run for run in runs if run.grid == 'pretrain')
    downstream = next(run for run in runs if run.grid == 'probe')
    return study, upstream, downstream


def test_queued_dependency_is_waiting() -> None:
    study, _, downstream = _runs()
    report = _report(
        run=downstream, state=StudyState.QUEUED, job_flags=frozenset({JobFlag.DEPENDENCY_PENDING})
    )

    (updated,) = apply_chain_states(
        study=study, reports=[report], upstream={}, followup=FollowupSettings()
    )

    assert updated.state is StudyState.WAITING


def test_never_started_run_with_failed_upstream_is_dependency_failed() -> None:
    study, upstream, downstream = _runs()
    report = _report(run=downstream, state=StudyState.MISSING)
    upstream_report = _report(run=upstream, state=StudyState.FAILED)

    (updated,) = apply_chain_states(
        study=study,
        reports=[report],
        upstream={upstream.run_path: upstream_report},
        followup=FollowupSettings(),
    )

    assert updated.state is StudyState.DEPENDENCY_FAILED


def test_cancelled_downstream_with_met_upstream_is_missing() -> None:
    study, upstream, downstream = _runs()
    report = _report(
        run=downstream, state=StudyState.FAILED, job_flags=frozenset({JobFlag.CANCELLED})
    )
    upstream_report = _report(run=upstream, state=StudyState.DONE)

    (updated,) = apply_chain_states(
        study=study,
        reports=[report],
        upstream={upstream.run_path: upstream_report},
        followup=FollowupSettings(),
    )

    assert updated.state is StudyState.MISSING


def test_dependency_never_flag_is_dependency_failed_without_upstream_reports() -> None:
    study, _, downstream = _runs()
    report = _report(
        run=downstream, state=StudyState.FAILED, job_flags=frozenset({JobFlag.DEPENDENCY_NEVER})
    )

    (updated,) = apply_chain_states(
        study=study, reports=[report], upstream={}, followup=FollowupSettings()
    )

    assert updated.state is StudyState.DEPENDENCY_FAILED


def test_four_grid_chain_propagates_dependency_failure_in_one_pass() -> None:
    study, runs = _deep_chain()
    reports = [
        _report(run=run, state=StudyState.FAILED if run.grid == 'a' else StudyState.MISSING)
        for run in reversed(runs)
    ]

    updated = apply_chain_states(
        study=study, reports=reports, upstream={}, followup=FollowupSettings()
    )

    states = {report.run.grid: report.state for report in updated}
    assert states == {
        'a': StudyState.FAILED,
        'b': StudyState.DEPENDENCY_FAILED,
        'c': StudyState.DEPENDENCY_FAILED,
        'd': StudyState.DEPENDENCY_FAILED,
    }


def _deep_chain() -> tuple[Study, list[StudyRun]]:
    config = _ChainConfig
    grids = {
        name: Grid(
            short_name=name,
            command=('python', '-m', name),
            config=config,
            axes={'dataset': {'a': {}}},
            after=() if upstream is None else (After(grid=upstream),),
            run_path='{grid}/{dataset}',
        )
        for name, upstream in (('d', 'c'), ('b', 'a'), ('a', None), ('c', 'b'))
    }
    study = Study(name='deep', grids=grids)
    return study, expand_runs(study=study)
