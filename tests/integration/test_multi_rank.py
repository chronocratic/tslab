import os
from pathlib import Path
import subprocess
import sys

import pytest

from tslab.core.artifact import Json
from tslab.core.status import read_status
from tslab.enums import InstanceState

SCRIPT = Path(__file__).parent / 'multi_rank_script.py'


def run_two_ranks(*, run_root: Path, logs_root: Path) -> tuple[list[int], list[str]]:
    logs_root.mkdir(parents=True)
    processes: list[subprocess.Popen[bytes]] = []
    log_paths = [logs_root / f'rank-{rank}.log' for rank in range(2)]
    shared_environment = {
        **os.environ,
        'SLURM_JOB_ID': '4242',
        'SLURM_JOB_NAME': 'tslab-test',
        'SLURM_NTASKS': '2',
        'SLURM_NTASKS_PER_NODE': '2',
        'SLURM_NODELIST': 'localhost',
        'SLURM_NNODES': '1',
        'SLURM_NODEID': '0',
    }
    for rank, log_path in enumerate(log_paths):
        environment = {**shared_environment, 'SLURM_PROCID': str(rank), 'SLURM_LOCALID': str(rank)}
        with log_path.open('wb') as output:
            processes.append(
                subprocess.Popen(  # noqa: S603 # why: launches this repository's test script
                    [sys.executable, str(SCRIPT), str(run_root)],
                    cwd=SCRIPT.parent,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            )
    try:
        codes = [process.wait(timeout=300) for process in processes]
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        for process in processes:
            process.wait()
        pytest.fail('two-rank training timed out')
    logs = [path.read_text(encoding='utf-8') for path in log_paths]
    return codes, logs


@pytest.mark.slow
def test_two_ranks_train_then_rank_zero_finishes(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    codes, logs = run_two_ranks(run_root=run_root, logs_root=tmp_path / 'first-logs')
    assert codes == [0, 0], '\n'.join(logs)
    snapshot = read_status(path=run_root / 'r' / 'status.json')
    assert snapshot is not None
    assert {record.state for record in snapshot.instances.values()} == {InstanceState.DONE}
    result = Json.load(directory=run_root / 'r' / 'after' / 'default')
    assert result.data == {'rank': '0'}


@pytest.mark.slow
def test_two_ranks_skip_cached_training_together(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    first_codes, _ = run_two_ranks(run_root=run_root, logs_root=tmp_path / 'first-logs')
    second_codes, logs = run_two_ranks(run_root=run_root, logs_root=tmp_path / 'second-logs')
    assert first_codes == [0, 0]
    assert second_codes == [0, 0], '\n'.join(logs)
    assert all('load train' in log for log in logs)
