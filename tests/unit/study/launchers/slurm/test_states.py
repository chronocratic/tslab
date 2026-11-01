"""SLURM state queries and cache updates."""

import pytest

from tslab.study.launchers.slurm import states


def test_query_states_returns_new_cache_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ended = {'1000_0': 'COMPLETED'}
    queried: list[list[str]] = []

    def query_job_states(*, job_ids: list[str]) -> dict[str, str]:
        queried.append(job_ids)
        return {'1000_1': 'RUNNING', '1000_2': 'COMPLETED'}

    monkeypatch.setattr(states, 'query_job_states', query_job_states)

    raw_states, new_cache = states._query_states(  # noqa: SLF001 # why: test cache immutability
        job_ids=['1000_0', '1000_1', '1000_2'], ended_states=ended
    )

    assert raw_states == {'1000_0': 'COMPLETED', '1000_1': 'RUNNING', '1000_2': 'COMPLETED'}
    assert new_cache == {'1000_0': 'COMPLETED', '1000_2': 'COMPLETED'}
    assert ended == {'1000_0': 'COMPLETED'}
    assert queried == [['1000_1', '1000_2']]
