"""Rank-zero cache decisions reach all SLURM tasks."""

import pytest

from tslab.ml.lightning import LightningTrainer
from tslab.ml.lightning.ranks import store_host


def test_agree_is_the_identity_in_a_single_task_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLURM_NTASKS', raising=False)
    assert LightningTrainer().agree_across_ranks(key='k', value=False) is False


def test_store_host_prefers_master_addr() -> None:
    environment = {'MASTER_ADDR': 'head', 'SLURM_JOB_NODELIST': 'node[3-5]'}
    assert store_host(environment=environment) == 'head'


def test_store_host_is_the_first_node_of_the_job() -> None:
    assert store_host(environment={'SLURM_JOB_NODELIST': 'node[03-05],other'}) == 'node03'
    assert store_host(environment={'SLURM_JOB_NODELIST': 'solo'}) == 'solo'
