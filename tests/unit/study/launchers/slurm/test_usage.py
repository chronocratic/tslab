"""SLURM accounting usage queries."""

from collections.abc import Mapping
import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, submit_study
from tslab.jobs.slurm import SlurmJobUsage
from tslab.study.launchers import JobUsage
import tslab.study.launchers.slurm.usage as slurm_usage
from tslab.study.reporting.failures import collect_failures
from tslab.study.reporting.state import StudyState


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


def test_oom_hint_without_numbers_when_sacct_fails(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
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
    with caplog.at_level('WARNING', logger='tslab.study.reporting.failures'):
        failures = collect_failures(context=ctx, states=(), lines=30)
    assert 'sacct' in caplog.text
    (failure,) = failures
    assert failure['state'] == StudyState.FAILED.value
    assert failure['hint'] == 'raise mem_per_cpu'
