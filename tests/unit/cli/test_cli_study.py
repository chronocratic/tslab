"""The ``tslab study`` command group: parsing, filters and pass-through."""

from pathlib import Path

import pytest

from tslab.cli import main
from tslab.cli.study import Submit


def test_cli_parses_filters_and_pass_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Submit] = []

    def run(self: Submit) -> int:
        seen.append(self)
        return 0

    monkeypatch.setattr(Submit, 'run', run)
    with pytest.raises(SystemExit) as exit_info:
        main(
            args=[
                'study',
                'submit',
                'toy_studies:study',
                '--machine',
                str(tmp_path / 'm.toml'),
                '--only',
                'recipe=a,b',
                '--only',
                'seed=42',
                '--grid',
                'g',
                '--dry-run',
                '--',
                '--run.rerun-from',
                'evaluate',
            ]
        )
    assert exit_info.value.code == 0
    assert len(seen) == 1
    assert seen[0].study == 'toy_studies:study'
    assert seen[0].machine == tmp_path / 'm.toml'
    assert seen[0].only == ('recipe=a,b', 'seed=42')
    assert seen[0].grid == ('g',)
    assert seen[0].include_done is False
    assert seen[0].dry_run is True
    assert seen[0].extra_args == ('--run.rerun-from', 'evaluate')


def test_pass_through_needs_a_command_that_takes_it() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['status', 'runs', 's', '--', '--x'])
    assert exit_info.value.code == 2


def test_help_lists_study_actions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', '--help'])
    assert exit_info.value.code == 0
    assert 'submit' in capsys.readouterr().out
