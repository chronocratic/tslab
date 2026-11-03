"""The analysis notebook of the project template: clean, compiles and marks synthetic data."""

import json
from pathlib import Path

from matplotlib.figure import Figure
import pytest

pytest.importorskip('pandas')
pytest.importorskip('matplotlib')
pytest.importorskip('scipy')

from tslab.analysis import Caption  # why: the extras must be checked first

REPOSITORY = Path(__file__).parents[2]
NOTEBOOK = REPOSITORY / 'src/tslab/templates/project/notebooks/analysis.ipynb'
SYNTHETIC_MARKER = '# synthetic'


def cells(*, kind: str) -> list[str]:
    """The sources of one cell type, with the placeholders replaced by ``demo``."""
    document = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [
        ''.join(cell['source'])
        .replace('{{package_name}}', 'demo')
        .replace('{{project_name}}', 'demo')
        for cell in document['cells']
        if cell['cell_type'] == kind
    ]


def test_notebook_is_clean_json() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert document['nbformat'] == 4
    assert all(
        cell.get('outputs', []) == [] and cell.get('execution_count') is None
        for cell in document['cells']
    )


def test_every_code_cell_compiles() -> None:
    for source in cells(kind='code'):
        compile(source, 'analysis.ipynb', 'exec')


def test_synthetic_sections_run_and_are_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shown: list[object] = []
    monkeypatch.setattr('tslab.analysis.show', lambda *, blocks: shown.extend(blocks))
    code = cells(kind='code')
    namespace: dict[str, object] = {}
    for source in [code[0], code[1], *(s for s in code if s.startswith(SYNTHETIC_MARKER))]:
        exec(source, namespace)  # noqa: S102 # why: runs the template cells under test
    captions = [block for block in shown if isinstance(block, Caption)]
    figures = [block for block in shown if isinstance(block, Figure)]
    assert captions
    assert all(caption.startswith('[SYNTHETIC DATA]') for caption in captions)
    assert figures
    assert all(any('SYNTHETIC DATA' in text.get_text() for text in f.texts) for f in figures)
    files = [path for path in (tmp_path / 'analysis').rglob('*') if path.is_file()]
    assert files
    assert all(path.is_relative_to(tmp_path / 'analysis/synthetic') for path in files)
    assert all(path.name.startswith('synthetic_') for path in files)
    assert not list(tmp_path.rglob('results_numbers.*'))


def test_synthetic_headings_say_so() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    headings = [
        ''.join(cell['source']).splitlines()[0]
        for cell in document['cells']
        if cell['cell_type'] == 'markdown' and ''.join(cell['source']).startswith(('## 3', '## 4'))
    ]
    assert headings
    assert all('Synthetic demo data' in heading for heading in headings)
