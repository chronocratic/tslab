"""Local CPU availability."""

from tslab.study.launchers.local.resources import usable_cpus


def test_usable_cpus_uses_given_affinity_query() -> None:
    assert usable_cpus(affinity=lambda: 3) == 3


def test_usable_cpus_returns_at_least_one() -> None:
    assert usable_cpus(affinity=lambda: 0) == 1
