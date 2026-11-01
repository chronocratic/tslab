"""The ``tslab study submit`` command: the messages of its run."""

import logging
from pathlib import Path

import pytest

from tests.unit.cli.study.helpers import toy_project
from tslab.cli.study import Submit


def test_dry_run_says_nothing_was_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Submit(study='toy_studies:study', machine='local.toml', dry_run=True).run()

    assert code == 0
    assert 'dry run: nothing submitted' in caplog.text
