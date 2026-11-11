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


def test_store_host_falls_back_to_the_node_list_of_the_step() -> None:
    assert store_host(environment={'SLURM_NODELIST': 'node[03-05]'}) == 'node03'


def test_store_host_is_the_local_host_without_any_node_list() -> None:
    assert store_host(environment={}) == '127.0.0.1'
    assert store_host(environment={'SLURM_JOB_NODELIST': ''}) == '127.0.0.1'
