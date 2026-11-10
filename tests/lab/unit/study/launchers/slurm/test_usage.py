"""SLURM accounting usage queries."""

from collections.abc import Mapping
import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from tests.lab.unit.study.helpers import context, submit_study
from tests.support.fake_slurm import FakeSlurm
from tslab.lab.enums import StudyState
from tslab.lab.jobs.slurm import SlurmJobUsage
from tslab.lab.study.launchers.common.types import JobUsage
import tslab.lab.study.launchers.slurm.usage as slurm_usage
from tslab.lab.study.reporting.failures import collect_failures


def test_usage_queries_all_selected_job_ids(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    expected = [report.job_id for report in ctx.reports if report.job_id is not None]
    returned: dict[str, SlurmJobUsage] = {
        expected[0]: SlurmJobUsage(
            state='COMPLETED',
            elapsed=timedelta(minutes=1),
            time_limit=timedelta(minutes=10),
            total_cpu=timedelta(seconds=30),
            max_rss=1024,
            alloc_cpus=2,
            gpus=0,
        )
    }
    calls: list[list[str]] = []

    def query_job_usage(*, job_ids: list[str]) -> Mapping[str, SlurmJobUsage]:
        calls.append(job_ids)
        return returned

    monkeypatch.setattr(slurm_usage, 'query_job_usage', query_job_usage)

    usage = slurm_usage.usage(context=ctx)

    assert calls == [expected]
    converted = usage[expected[0]]
    assert isinstance(converted, JobUsage)
    assert dataclasses.asdict(converted) == dataclasses.asdict(returned[expected[0]])


def test_failure_usage_queries_failed_runs_too(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    reports = [
        dataclasses.replace(ctx.reports[0], job_state='FAILED'),
        dataclasses.replace(ctx.reports[1], job_state='OUT_OF_MEMORY'),
        dataclasses.replace(ctx.reports[2], job_state='TIMEOUT'),
    ]
    calls: list[list[str]] = []

    def query_job_usage(*, job_ids: list[str]) -> Mapping[str, SlurmJobUsage]:
        calls.append(job_ids)
        return dict[str, SlurmJobUsage]()

    monkeypatch.setattr(slurm_usage, 'query_job_usage', query_job_usage)
    slurm_usage.failure_usage(context=ctx, reports=reports)
    assert calls == [[reports[0].job_id, reports[1].job_id, reports[2].job_id]]


def test_failure_usage_reads_the_exit_code_of_a_completed_job(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    (report,) = [dataclasses.replace(ctx.reports[0], job_state='COMPLETED')]

    def query_job_usage(*, job_ids: list[str]) -> Mapping[str, SlurmJobUsage]:
        return {
            job_ids[0]: SlurmJobUsage(
                state='COMPLETED',
                elapsed=timedelta(minutes=1),
                time_limit=timedelta(minutes=10),
                total_cpu=timedelta(seconds=30),
                max_rss=1024,
                alloc_cpus=2,
                gpus=0,
                exit_code='1:137',
            )
        }

    monkeypatch.setattr(slurm_usage, 'query_job_usage', query_job_usage)
    usage = slurm_usage.failure_usage(context=ctx, reports=[report])
    assert report.job_id is not None

    assert usage[report.job_id].exit_code == '1:137'


def test_failure_usage_reads_the_exit_code_of_an_unknown_state_job(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    (report,) = [dataclasses.replace(ctx.reports[0], job_state=None)]

    def query_job_usage(*, job_ids: list[str]) -> Mapping[str, SlurmJobUsage]:
        return {
            job_ids[0]: SlurmJobUsage(
                state='',
                elapsed=timedelta(minutes=1),
                time_limit=timedelta(minutes=10),
                total_cpu=timedelta(seconds=30),
                max_rss=1024,
                alloc_cpus=2,
                gpus=0,
                exit_code='2:143',
            )
        }

    monkeypatch.setattr(slurm_usage, 'query_job_usage', query_job_usage)
    usage = slurm_usage.failure_usage(context=ctx, reports=[report])
    assert report.job_id is not None

    assert usage[report.job_id].exit_code == '2:143'


def test_oom_hint_without_numbers_when_sacct_fails(
    project: Path,
    fake: FakeSlurm,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('tslab.lab.jobs.slurm.commands._sleep', lambda _: None)
    submit_study(project=project)
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_0|OUT_OF_MEMORY\n1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'
                ),
            },
            {'exit': 1, 'stderr': 'sacct: error: down'},
        ],
    )
    ctx = context(project=project, machine_file='cluster.toml')
    with caplog.at_level('WARNING', logger='tslab.lab.study.reporting.failures'):
        failures = collect_failures(context=ctx, states=(), lines=30)
    assert 'sacct' in caplog.text
    (failure,) = failures
    assert failure['state'] == StudyState.FAILED.value
    assert failure['hint'] == 'raise mem_per_cpu'
