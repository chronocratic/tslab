"""SLURM script requeue modes."""

from tslab.study.launchers.slurm.requeue_mode import RequeueMode


def test_requeue_mode_parses_all_values() -> None:
    assert [RequeueMode(value) for value in ('disabled', 'enabled', 'expedite')] == [
        RequeueMode.DISABLED,
        RequeueMode.ENABLED,
        RequeueMode.EXPEDITE,
    ]
