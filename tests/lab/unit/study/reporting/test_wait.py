"""The study wait loop, using injected clocks and sleep."""

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from tests.lab.unit.study.helpers import context, submit_study
from tests.support.fake_backend import FakeBackend
from tslab.lab.cli import main
from tslab.lab.enums import JobPhase, StudyState
from tslab.lab.study.context import Context
from tslab.lab.study.reporting.outcome import EXIT_TIMEOUT
from tslab.lab.study.reporting.wait import wait_for_study


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


def test_wait_returns_0_when_all_done_after_two_checks(project: Path) -> None:
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


def test_wait_stops_on_first_failure_without_keep_going(project: Path) -> None:
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


def test_wait_keep_going_waits_until_nothing_live(project: Path) -> None:
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


def test_wait_times_out_with_124(project: Path) -> None:
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
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
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
    with caplog.at_level(logging.WARNING, logger='tslab.lab.study.reporting.wait'):
        code, _, _ = wait_for_study(
            load=prepared.__next__,
            interval=10,
            timeout=None,
            keep_going=False,
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
            clock=lambda: elapsed[0],
        )
    assert code == 0
    assert caplog.text.count('job states unknown; still waiting') == 1


def test_wait_returns_last_state_when_reload_is_interrupted(project: Path) -> None:
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


def test_wait_logs_only_when_counts_change(project: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = context(project=project, machine_file='local.toml')
    running = _with_states(context=base, states=(StudyState.RUNNING,) * len(base.reports))
    done = _with_states(context=base, states=(StudyState.DONE,) * len(base.reports))
    prepared = iter([running, running, done])
    elapsed = [0.0]
    with caplog.at_level(logging.INFO, logger='tslab.lab.study.reporting.wait'):
        wait_for_study(
            load=prepared.__next__,
            interval=10,
            timeout=None,
            keep_going=False,
            sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
            clock=lambda: elapsed[0],
        )
    logger = 'tslab.lab.study.reporting.wait'
    lines = [record.getMessage() for record in caplog.records if record.name == logger]
    assert len(lines) == 2
    assert 'running 4' in lines[0]
    assert 'done 4' in lines[1]


def test_wait_gap_doubles_up_to_interval(project: Path) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    done = (StudyState.DONE,) * len(base.reports)
    code, _, waited, sleeps = _wait(base=base, sequence=(running,) * 6 + (done,), interval=300)
    assert code == 0
    assert sleeps == [30, 60, 120, 240, 300, 300]
    assert waited == 1050


def test_wait_gap_resets_when_counts_change(project: Path) -> None:
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


def test_wait_gap_never_exceeds_small_interval(project: Path) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    done = (StudyState.DONE,) * len(base.reports)
    code, _, waited, sleeps = _wait(
        base=base, sequence=(running, running, running, done), interval=10
    )
    assert code == 0
    assert sleeps == [10, 10, 10]
    assert waited == 30


def test_wait_timeout_uses_current_gap(project: Path) -> None:
    base = context(project=project, machine_file='local.toml')
    running = (StudyState.RUNNING,) * len(base.reports)
    code, _, waited, sleeps = _wait(
        base=base, sequence=(running, running, running), interval=300, timeout=100
    )
    assert code == EXIT_TIMEOUT
    assert sleeps == [30, 60]
    assert waited == 90


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
    project: Path, fake_backend: FakeBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.QUEUED})
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


def test_wait_rejects_empty_selection(project: Path, fake_backend: FakeBackend) -> None:
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
