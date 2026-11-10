import pytest

from tslab.core.ranks import is_writer_process


def test_is_writer_process_is_true_when_all_rank_variables_are_zero_or_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('SLURM_PROCID', raising=False)
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    assert is_writer_process()


def test_is_writer_process_is_false_on_a_non_zero_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ('RANK', 'SLURM_PROCID', 'LOCAL_RANK'):
        monkeypatch.setenv(variable, '1')
        assert not is_writer_process()
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('SLURM_PROCID', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    assert is_writer_process()
