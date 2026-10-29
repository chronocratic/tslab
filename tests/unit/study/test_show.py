"""Study description documents and human output."""

import json
from pathlib import Path
from typing import cast

import pytest

from tslab.cli import main
from tslab.helpers.text import TextTable
from tslab.study.definition import expand_runs
from tslab.study.documents import show_document
from tslab.study.loading import load_study


def test_show_document_lists_axes_and_counts(project: Path) -> None:
    study = load_study(reference='toy_studies:study')
    document = show_document(study=study)
    grids = cast('list[dict[str, object]]', document['grids'])
    assert document['schema'] == 'tslab.study-show/1'
    assert grids[0]['axes'] == {'model': ['small', 'large'], 'seed': ['1', '2']}
    assert grids[0]['runs'] == len(expand_runs(study=study))
    assert document['total_runs'] == len(expand_runs(study=study))


def test_show_cli_prints_table_without_machine(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'show', 'toy_studies:study'])
    assert raised.value.code == 0
    tables = [getattr(record, 'tslab_table', None) for record in caplog.records]
    assert any(
        isinstance(table, TextTable) and table.header == ('grid', 'short', 'runs', 'axes')
        for table in tables
    )
    assert 'command of g:' in caplog.text
    assert 'select with --grid <grid> or --only <axis>=<value>' in caplog.text


def test_show_cli_prints_json(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'show', 'toy_studies:study', '--json'])
    assert raised.value.code == 0
    assert json.loads(capsys.readouterr().out)['schema'] == 'tslab.study-show/1'
