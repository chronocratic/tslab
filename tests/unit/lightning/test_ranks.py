"""Rank-zero cache decisions reach all SLURM tasks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tslab.lightning import LightningTrainer


def test_agree_is_the_identity_in_a_single_task_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SLURM_NTASKS', raising=False)
    assert LightningTrainer().agree_across_ranks(key='k', value=False) is False


@pytest.mark.slow
def test_agree_returns_the_value_of_rank_zero(tmp_path: Path) -> None:
    code = (
        'import os; from tslab.lightning.ranks import agree; '
        'print(agree(key="k", value=os.environ["SLURM_PROCID"] == "0", environment=os.environ))'
    )
    environment = {**os.environ, 'SLURM_JOB_ID': '4242', 'SLURM_NTASKS': '2'}
    processes = [
        subprocess.Popen(  # noqa: S603 # why: execute two independent rank interpreters
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            env={**environment, 'SLURM_PROCID': str(rank)},
        )
        for rank in (0, 1)
    ]
    results = {rank: process.communicate(timeout=60) for rank, process in enumerate(processes)}
    assert [process.returncode for process in processes] == [0, 0], results
    assert {rank: output.strip() == 'True' for rank, (output, _) in results.items()} == {
        0: True,
        1: True,
    }
