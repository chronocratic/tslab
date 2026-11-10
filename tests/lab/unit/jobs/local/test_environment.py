"""Tests for the environment of local tasks."""

from tslab.lab.jobs.local.environment import without_scheduler_variables


def test_scheduler_variables_are_dropped() -> None:
    environ = {'SLURM_NTASKS': '4', 'SLURM_PROCID': '0', 'PATH': '/bin', 'MY_SLURM_X': '1'}
    assert without_scheduler_variables(environ=environ) == {'PATH': '/bin', 'MY_SLURM_X': '1'}
