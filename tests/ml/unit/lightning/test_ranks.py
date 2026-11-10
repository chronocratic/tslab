"""Rank-zero cache decisions reach all SLURM tasks."""

import pytest

from tslab.lightning import LightningTrainer


def test_agree_is_the_identity_in_a_single_task_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLURM_NTASKS', raising=False)
    assert LightningTrainer().agree_across_ranks(key='k', value=False) is False
