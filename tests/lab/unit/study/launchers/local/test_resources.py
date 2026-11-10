"""Local CPU availability."""

from tslab.lab.study.launchers.local.resources import usable_cpus


def test_usable_cpus_returns_at_least_one() -> None:
    assert usable_cpus(affinity=lambda: 0) == 1
