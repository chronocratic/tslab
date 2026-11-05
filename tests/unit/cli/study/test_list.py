"""The ``tslab study list`` command and short-name failures."""

import json
import logging
from pathlib import Path

import pytest

from tslab.cli.common import UsageError
from tslab.cli.study import List, Show


def _write_studies(*, project: Path, package: str, missing_quick_study: bool = False) -> None:
    """Write two discoverable studies and their shared package."""
    studies = project / package / 'studies'
    for folder in (project / package, studies, studies / 'common', studies / 'quick'):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / '__init__.py').write_text('', encoding='utf-8')
    (project / package / '__init__.py').write_text('', encoding='utf-8')
    (studies / '_private.py').write_text('', encoding='utf-8')
    (studies / 'full.py').write_text(
        'from tests.unit.study.configs import grid\n'
        'from tslab.study import Study\n'
        "study = Study(name='loaded-full', grids={'g': grid()})\n",
        encoding='utf-8',
    )
    quick_source = (
        ''
        if missing_quick_study
        else (
            'from tests.unit.study.configs import grid\n'
            'from tslab.study import Study\n'
            "study = Study(name='loaded-quick', grids={'g': grid()})\n"
        )
    )
    (studies / 'quick' / '__init__.py').write_text(quick_source, encoding='utf-8')
    (project / 'tslab.toml').write_text(f'studies = "{package}.studies"\n', encoding='utf-8')


def _package_name(*, tmp_path: Path) -> str:
    return f'pkg_{tmp_path.name.replace("-", "_")}'


def test_list_prints_short_names_and_study_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    package = _package_name(tmp_path=tmp_path)
    _write_studies(project=tmp_path, package=package)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.INFO):
        assert List().run() == 0

    assert [record.getMessage() for record in caplog.records] == [
        'full  loaded-full',
        'quick  loaded-quick',
    ]


def test_list_json_has_schema_and_study_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package_name(tmp_path=tmp_path)
    _write_studies(project=tmp_path, package=package)
    monkeypatch.chdir(tmp_path)

    assert List(json=True).run() == 0

    assert json.loads(capsys.readouterr().out) == {
        'schema': 'tslab.study-list/1',
        'studies': [
            {'short_name': 'full', 'name': 'loaded-full'},
            {'short_name': 'quick', 'name': 'loaded-quick'},
        ],
    }


def test_missing_short_name_lists_available_studies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package_name(tmp_path=tmp_path)
    _write_studies(project=tmp_path, package=package)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        UsageError,
        match=(
            rf"no study 'nope': cannot import {package}\.studies\.nope; "
            r'studies found: full, quick'
        ),
    ):
        Show(study='nope').run()


def test_short_name_module_without_study_reports_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package_name(tmp_path=tmp_path)
    _write_studies(project=tmp_path, package=package, missing_quick_study=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        UsageError,
        match=(
            rf"no study 'quick': {package}\.studies\.quick has no attribute 'study'; "
            r'studies found: full, quick'
        ),
    ):
        Show(study='quick').run()
