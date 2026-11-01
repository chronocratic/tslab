"""The study wait loop, using injected clocks and sleep."""

import dataclasses
import json
import logging
import os
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, jobs_of, submit_study
from tslab.cli import main
from tslab.cli.study import Wait
from tslab.study.context import Context, context_for, load_context, Selection
from tslab.study.definition import load_study
from tslab.study.launchers.slurm.states import slurm_task_ids
from tslab.study.reporting.outcome import EXIT_TIMEOUT
from tslab.study.reporting.state import StudyState
from tslab.study.reporting.wait import wait_for_study


def _with_states(
    *, context: Context, states: tuple[StudyState, ...], job_states_known: bool | None = None
) -> Context:
    reports = [
        dataclasses.replace(report, state=state, reason=None)
        for report, state in zip(context.reports, states, strict=True)
    ]
    return dataclasses.replace(
        context,
        reports=reports,
        job_states=dataclasses.replace(
            context.job_states,
            known=context.job_states.known if job_states_known is None else job_states_known,
        ),
    )


def _wait(
    *,
    base: Context,
    sequence: tuple[tuple[StudyState, ...], ...],
    interval: float,
    timeout: float | None = None,
    keep_going: bool = False,
) -> tuple[int, Context, float, list[float]]:
    """Run wait_for_study over the states of each check, recording the sleeps."""
    elapsed = [0.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        elapsed[0] += seconds

    code, last, waited = wait_for_study(
        load=iter(_sequence(base=base, sequence=sequence)).__next__,
        interval=interval,
        timeout=timeout,
        keep_going=keep_going,
        sleep=fake_sleep,
        clock=lambda: elapsed[0],
    )
    return code, last, waited, sleeps


def _sequence(*, base: Context, sequence: tuple[tuple[StudyState, ...], ...]) -> list[Context]:
    return [_with_states(context=base, states=states) for states in sequence]


def test_wait_returns_0_when_all_done_after_two_checks(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    prepared = iter(
        [
            _with_states(context=base, states=(StudyState.RUNNING,) * len(base.reports)),
            _with_states(context=base, states=(StudyState.DONE,) * len(base.reports)),
        ]
    )
    elapsed = [0.0]
    code, last, waited = wait_for_study(
        load=prepared.__next__,
        interval=10,
        timeout=None,
        keep_going=False,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        clock=lambda: elapsed[0],
    )
    assert code == 0
    assert all(report.state is StudyState.DONE for report in last.reports)
    assert waited == 10


def test_wait_stops_on_first_failure_without_keep_going(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    states = (StudyState.FAILED, StudyState.RUNNING, StudyState.DONE, StudyState.DONE)
    code, _, waited = wait_for_study(
        load=lambda: _with_states(context=base, states=states),
        interval=10,
        timeout=None,
        keep_going=False,
        sleep=lambda _: pytest.fail('wait stopped without sleeping'),
        clock=lambda: 0,
    )
    assert code == 4
    assert waited == 0


def test_wait_keep_going_waits_until_nothing_live(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    prepared = iter(
        [
            _with_states(
                context=base,
                states=(StudyState.FAILED, StudyState.RUNNING, StudyState.DONE, StudyState.DONE),
            ),
            _with_states(
                context=base,
                states=(StudyState.FAILED, StudyState.DONE, StudyState.DONE, StudyState.DONE),
            ),
        ]
    )
    elapsed = [0.0]
    code, _, waited = wait_for_study(
        load=prepared.__next__,
        interval=10,
        timeout=None,
        keep_going=True,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        clock=lambda: elapsed[0],
    )
    assert code == 4
    assert waited == 10


def test_wait_times_out_with_124(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = _with_states(context=base, states=(StudyState.RUNNING,) * len(base.reports))
    elapsed = [0.0]
    code, _, waited = wait_for_study(
        load=lambda: running,
        interval=10,
        timeout=15,
        keep_going=False,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        clock=lambda: elapsed[0],
    )
    assert code == 124
    assert waited == 10


def test_wait_keeps_going_through_unknown_states_and_warns_once(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    base = context(project=project, machine_file='cluster.toml')
    unknown = _with_states(
        context=base, states=(StudyState.MISSING,) * len(base.reports), job_states_known=False
    )
    done = _with_states(
        context=base, states=(StudyState.DONE,) * len(base.reports), job_states_known=True
    )
    prepared = iter([unknown, unknown, unknown, done])
    elapsed = [0.0]
    with caplog.at_level(logging.WARNING, logger='tslab.study.reporting.wait'):
        code, _, _ = wait_for_study(
            load=prepared.__next__,
            interval=10,
            timeout=None,
            keep_going=False,
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
            clock=lambda: elapsed[0],
        )
    assert code == 0
    assert caplog.text.count('SLURM job states unknown; still waiting') == 1


def test_wait_returns_last_state_when_reload_is_interrupted(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = _with_states(context=base, states=(StudyState.RUNNING,) * len(base.reports))
    checks = iter([running])

    def load() -> Context:
        try:
            return next(checks)
        except StopIteration as error:
            raise KeyboardInterrupt from error

    code, last, waited = wait_for_study(
        load=load,
        interval=10,
        timeout=None,
        keep_going=False,
        sleep=lambda _: None,
        clock=lambda: 0,
    )

    assert code == 3
    assert last is running
    assert waited == 0


def test_wait_logs_only_when_counts_change(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    base = context(project=project, machine_file='local.toml')
    running = _with_states(context=base, states=(StudyState.RUNNING,) * len(base.reports))
    done = _with_states(context=base, states=(StudyState.DONE,) * len(base.reports))
    prepared = iter([running, running, done])
    elapsed = [0.0]
    with caplog.at_level(logging.INFO, logger='tslab.study.reporting.wait'):
        wait_for_study(
            load=prepared.__next__,
            interval=10,
            timeout=None,
            keep_going=False,
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
            clock=lambda: elapsed[0],
        )
    logger = 'tslab.study.reporting.wait'
    lines = [record.getMessage() for record in caplog.records if record.name == logger]
    assert len(lines) == 2
    assert 'running 4' in lines[0]
    assert 'done 4' in lines[1]


def test_wait_gap_doubles_up_to_interval(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    done = (StudyState.DONE,) * len(base.reports)
    code, _, waited, sleeps = _wait(base=base, sequence=(running,) * 6 + (done,), interval=300)
    assert code == 0
    assert sleeps == [30, 60, 120, 240, 300, 300]
    assert waited == 1050


def test_wait_gap_resets_when_counts_change(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    one_done = (StudyState.DONE, *running[1:])
    done = (StudyState.DONE,) * len(base.reports)
    code, _, waited, sleeps = _wait(
        base=base, sequence=(running, running, running, one_done, one_done, done), interval=300
    )
    assert code == 0
    assert sleeps == [30, 60, 120, 30, 60]
    assert waited == 300


def test_wait_gap_never_exceeds_small_interval(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    done = (StudyState.DONE,) * len(base.reports)
    code, _, waited, sleeps = _wait(
        base=base, sequence=(running, running, running, done), interval=10
    )
    assert code == 0
    assert sleeps == [10, 10, 10]
    assert waited == 30


def test_wait_timeout_uses_current_gap(project: Path, fake: FakeSlurm) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    code, _, waited, sleeps = _wait(
        base=base, sequence=(running, running, running), interval=300, timeout=100
    )
    assert code == EXIT_TIMEOUT
    assert sleeps == [30, 60]
    assert waited == 90


def test_wait_cli_default_interval_is_300() -> None:
    assert Wait(study='s', machine='m').interval == 300.0


def test_wait_rejects_interval_below_10(project: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'wait',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--interval',
                '5',
            ]
        )
    assert raised.value.code == 2


def test_wait_json_reports_timeout(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'wait',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--interval',
                '10',
                '--timeout',
                '0',
                '--json',
            ]
        )
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 124
    assert document['schema'] == 'tslab.study-wait/1'
    assert document['exit_code'] == 124
    assert document['timed_out'] is True


def test_wait_rejects_empty_selection(project: Path, fake: FakeSlurm) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'wait',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--grid',
                'missing',
            ]
        )
    assert raised.value.code == 2


def _query_ids(call: list[str]) -> list[str]:
    return call[call.index('--jobs') + 1].split(',')


def test_wait_stops_asking_slurm_about_ended_tasks(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    ids = slurm_task_ids(jobs=jobs_of(project=project))
    first = '\n'.join(f'{item}|{"COMPLETED" if item == ids[0] else "RUNNING"}' for item in ids)
    second = '\n'.join(f'{item}|COMPLETED' for item in ids)
    prior = len(fake.calls('sacct'))
    fake.set_answers(
        'sacct', [{'stdout': first + '\n'}] * (prior + 1) + [{'stdout': second + '\n'}]
    )
    ended: dict[str, str] = {}
    for _ in range(2):
        load_context(
            reference='toy_studies:study',
            machine_path=project / 'cluster.toml',
            selection=Selection(),
            project_root=Path.cwd(),
            local_cpus=1,
            ended_states=ended,
            environment=os.environ,
        )
    queries = fake.calls('sacct')[prior:]
    assert _query_ids(queries[0]) == ids
    assert _query_ids(queries[1]) == ids[1:]
    assert ended == dict.fromkeys(ids, 'COMPLETED')


def test_failed_query_leaves_the_cache_unchanged(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'down'}])
    ended: dict[str, str] = {}
    context_for(
        study=load_study(reference='toy_studies:study'),
        machine_path=project / 'cluster.toml',
        selection=Selection(),
        project_root=Path.cwd(),
        local_cpus=1,
        ended_states=ended,
        environment=os.environ,
    )
    assert ended == {}
