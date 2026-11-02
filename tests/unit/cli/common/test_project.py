"""The request resolution: the short study name and the machine by host."""

from pathlib import Path

import pytest

from tslab.cli.common import resolve_request
from tslab.cli.common.errors import UsageError
from tslab.study.project import PROJECT_FILE


def _tslab_toml(*, tmp_path: Path, text: str) -> None:
    (tmp_path / PROJECT_FILE).write_text(text, encoding='utf-8')


def test_flag_wins_over_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _tslab_toml(tmp_path=tmp_path, text='[machine_by_host]\n"*" = "table"\n')
    request = resolve_request(study='pkg:study', machine='flag', start=tmp_path, host='laptop')
    assert request.machine == 'flag'
    assert capsys.readouterr().err == ''


def test_host_pattern_picks_and_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _tslab_toml(tmp_path=tmp_path, text='[machine_by_host]\n"login*" = "slurm"\n')
    request = resolve_request(study='pkg:study', machine=None, start=tmp_path, host='login1')
    assert request.machine == 'slurm'
    pattern = 'login*'
    line = f'machine slurm from {tmp_path / PROJECT_FILE} (host login1 matches {pattern!r})'
    assert line in capsys.readouterr().err


def test_no_match_is_a_usage_error_naming_host_and_patterns(tmp_path: Path) -> None:
    _tslab_toml(tmp_path=tmp_path, text='[machine_by_host]\n"login*" = "slurm"\n')
    with pytest.raises(UsageError, match='laptop') as raised:
        resolve_request(study='pkg:study', machine=None, start=tmp_path, host='laptop')
    assert "'login*'" in str(raised.value)


def test_no_table_is_a_usage_error_saying_so(tmp_path: Path) -> None:
    _tslab_toml(tmp_path=tmp_path, text='studies = "pkg"\n')
    with pytest.raises(UsageError, match=r'\[machine_by_host\]'):
        resolve_request(study='pkg:study', machine=None, start=tmp_path, host='laptop')


def test_empty_table_is_a_usage_error_saying_the_table_is_empty(tmp_path: Path) -> None:
    _tslab_toml(tmp_path=tmp_path, text='[machine_by_host]\n')
    with pytest.raises(UsageError, match='laptop') as raised:
        resolve_request(study='pkg:study', machine=None, start=tmp_path, host='laptop')
    assert 'has no pattern' in str(raised.value)
    assert str(tmp_path / PROJECT_FILE) in str(raised.value)


def test_no_file_and_no_flag_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match=r'tslab\.toml'):
        resolve_request(study='pkg:study', machine=None, start=tmp_path, host='laptop')


def test_broken_file_fails_even_with_a_flag(tmp_path: Path) -> None:
    _tslab_toml(tmp_path=tmp_path, text='studies = 3\n')
    with pytest.raises(UsageError, match=r'tslab\.toml'):
        resolve_request(study='pkg:study', machine='flag', start=tmp_path, host='laptop')


@pytest.mark.parametrize(
    ('text', 'typed', 'expected'),
    [
        ('studies = "pkg"\n', 'ablation', 'pkg:ablation'),
        ('studies = "pkg"\n', 'other:x', 'other:x'),
        ('[machine_by_host]\n"*" = "m"\n', 'ablation', 'ablation'),
        (None, 'ablation', 'ablation'),
    ],
)
def test_short_name_is_expanded(
    tmp_path: Path, text: str | None, typed: str, expected: str
) -> None:
    if text is not None:
        _tslab_toml(tmp_path=tmp_path, text=text)
    request = resolve_request(study=typed, machine='flag', start=tmp_path, host='laptop')
    assert request.study == expected


def test_no_match_is_fine_when_no_machine_is_needed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tslab_toml(tmp_path=tmp_path, text='[machine_by_host]\n"login*" = "slurm"\n')
    request = resolve_request(
        study='pkg:study', machine=None, start=tmp_path, host='laptop', needs_machine=False
    )
    assert request.machine is None
    assert capsys.readouterr().err == ''
