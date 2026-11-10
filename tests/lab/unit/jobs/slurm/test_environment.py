import pytest

from tslab.lab.jobs.slurm import task_count


@pytest.mark.parametrize(('environment', 'expected'), [({'SLURM_NTASKS': '4'}, 4), ({}, 1)])
def test_task_count_reads_ntasks(*, environment: dict[str, str], expected: int) -> None:
    assert task_count(environment=environment) == expected
