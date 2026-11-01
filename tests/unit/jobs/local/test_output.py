from pathlib import Path

from tslab.jobs.local import tail_lines


def test_tail_lines_returns_last_lines(*, tmp_path: Path) -> None:
    path = tmp_path / 'launch.log'
    path.write_text('first\nsecond\nthird\nfourth\n', encoding='utf-8')
    assert tail_lines(path=path, lines=2) == ['third', 'fourth']
    assert tail_lines(path=path, lines=10) == ['first', 'second', 'third', 'fourth']


def test_tail_lines_of_missing_file_is_none(*, tmp_path: Path) -> None:
    assert tail_lines(path=tmp_path / 'missing.log', lines=5) is None
