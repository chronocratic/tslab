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
