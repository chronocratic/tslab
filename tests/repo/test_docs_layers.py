"""Check the docs against the layer split: the lab docs stay neutral and seeding is explained."""

from pathlib import Path
import re

ROOT = Path(__file__).parents[2]
DOCS = ROOT / 'docs'
FORBIDDEN = re.compile(r'(?i)forecast|encoder|\bprobes?\b|horizon|time series')
SEEDING_PAGES = (
    DOCS / 'lab' / 'run-options.md',
    DOCS / 'ml' / 'index.md',
    DOCS / 'lab' / 'guide.md',
    DOCS / 'lab' / 'agents.md',
    ROOT / 'src' / 'tslab' / 'lab' / 'templates' / 'project' / 'AGENTS.md',
    ROOT / 'src' / 'tslab' / 'ts' / 'templates' / 'project' / 'AGENTS.md',
)


def test_lab_docs_have_no_time_series_vocabulary() -> None:
    hits = [
        f'{path}:{number}'
        for path in (DOCS / 'lab').rglob('*.md')
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
        if FORBIDDEN.search(line)
    ]

    assert hits == []


def test_seeding_is_documented_everywhere() -> None:
    for path in SEEDING_PAGES:
        text = path.read_text(encoding='utf-8')
        assert 'tslab.ml' in text, path
        assert 'seed' in text, path


def test_old_top_level_docs_are_gone() -> None:
    names = {path.name for path in DOCS.glob('*.md')}

    assert names == {'index.md', 'changelog.md', 'contributing.md'}
