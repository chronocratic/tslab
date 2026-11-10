"""The ``tslab doctor`` command."""

import json
import logging
from pathlib import Path

import pytest

from tslab.cli import main
from tslab.core.meta import hostname
from tslab.study.project import PROJECT_FILE


def _write_profile(*, tmp_path: Path) -> Path:
    profile = tmp_path / 'local.toml'
    profile.write_text('launcher = "local"\noutput_root = "runs"\n', encoding='utf-8')
    return profile


def test_doctor_json_shape_and_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = tmp_path / 'local.toml'
    profile.write_text('launcher = "local"\noutput_root = "runs"\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--machine', str(profile), '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['schema'] == 'tslab.doctor/1'
    assert document['machine'] == profile.resolve().as_posix()
    assert document['checks'][0]['name'] == 'profile'


def test_doctor_reports_no_project_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = _write_profile(tmp_path=tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--machine', str(profile), '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['project_settings'] == {
        'path': None,
        'values': {
            'studies': {'value': None, 'source': 'default'},
            'machine_by_host': {'value': str(profile), 'source': 'flag'},
            'runs': {'value': None, 'source': 'default'},
        },
    }


def test_doctor_reports_file_values_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = _write_profile(tmp_path=tmp_path)
    (tmp_path / 'tslab.toml').write_text(
        'studies = "my_paper.studies"\nruns = "runs"\n', encoding='utf-8'
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--machine', str(profile), '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    project_settings = document['project_settings']
    assert project_settings['path'] == (tmp_path / 'tslab.toml').as_posix()
    values = project_settings['values']
    assert values['studies'] == {'value': 'my_paper.studies', 'source': 'file'}
    assert values['machine_by_host'] == {'value': str(profile), 'source': 'flag'}
    assert values['runs'] == {'value': (tmp_path / 'runs').as_posix(), 'source': 'file'}


def test_doctor_reads_the_settings_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = _write_profile(tmp_path=tmp_path)
    (tmp_path / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    reads = 0
    original_read_text = Path.read_text

    def _counting_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        nonlocal reads
        if self.name == PROJECT_FILE:
            reads += 1
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, 'read_text', _counting_read_text)
    with pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--machine', str(profile), '--json'])
    assert raised.value.code == 0
    assert reads == 1


def test_doctor_picks_the_machine_by_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = _write_profile(tmp_path=tmp_path)
    (tmp_path / 'tslab.toml').write_text(
        f'[machine_by_host]\n"*" = "{profile}"\n', encoding='utf-8'
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--json'])
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert raised.value.code == 0
    assert document['machine'] == profile.resolve().as_posix()
    line = f"machine {profile} from {tmp_path / 'tslab.toml'} (host {hostname()} matches '*')"
    assert line in captured.err


def test_doctor_text_lists_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    profile = _write_profile(tmp_path=tmp_path)
    (tmp_path / 'tslab.toml').write_text(
        'studies = "my_paper.studies"\nruns = "runs"\n', encoding='utf-8'
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as raised:
        main(args=['doctor', '--machine', str(profile)])
    assert raised.value.code == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any('project settings:' in message for message in messages)
    table = next(message for message in messages if 'machine_by_host' in message)
    for key in ('studies', 'machine_by_host', 'runs'):
        assert key in table
