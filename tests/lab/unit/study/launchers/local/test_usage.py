"""Usage files from local tasks."""

from pathlib import Path

from tests.unit.study.helpers import context
from tslab.study.launchers.local.usage import usage


def test_usage_without_recorded_local_tasks_is_empty(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    assert usage(context=ctx) == ({}, {})
