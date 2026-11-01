"""SLURM accounting usage queries."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, submit_study
from tslab.jobs.slurm import SlurmJobUsage
import tslab.study.launchers.slurm.usage as slurm_usage


def test_usage_queries_all_selected_job_ids(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    returned: dict[str, SlurmJobUsage] = {}
    calls: list[list[str]] = []

    def query_job_usage(*, job_ids: list[str]) -> Mapping[str, SlurmJobUsage]:
        calls.append(job_ids)
        return returned

    monkeypatch.setattr(slurm_usage, 'query_job_usage', query_job_usage)

    usage = slurm_usage.usage(context=ctx)

    expected = [report.job_id for report in ctx.reports if report.job_id is not None]
    assert calls == [expected]
    assert usage is returned
