"""Run the experiment once and dry-run the study, so a broken setup shows in seconds."""

from pathlib import Path

import pytest
import tslab.lab
from tslab.lab.cli import main as tslab_main
from tslab.lab.core.status import read_status

from {{package_name}}.experiment import ExperimentConfig, main


def test_experiment_runs(tmp_path: Path) -> None:
    config = ExperimentConfig(output_root=tmp_path, scale=2.0)

    main(config=config)

    snapshot = read_status(path=tmp_path / config.run_path / 'status.json')
    assert snapshot is not None
    states = {record.state for record in snapshot.instances.values()}
    assert states == {tslab.lab.InstanceState.DONE}


def test_example_study_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.setenv('UV_PROJECT', str(root))
    machine = str(root / 'machines' / 'local.toml')
    (tmp_path / 'tslab.toml').write_text(
        (root / 'tslab.toml').read_text(encoding='utf-8'), encoding='utf-8'
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as raised:
        tslab_main(args=['study', 'submit', 'example', '--machine', machine, '--dry-run'])

    assert raised.value.code == 0
