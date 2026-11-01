"""The ``tslab doctor`` command."""

import json
from pathlib import Path

import pytest

from tslab.cli import main


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
