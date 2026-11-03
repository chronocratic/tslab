import logging
from pathlib import Path
import re

from matplotlib.figure import Figure
import pandas as pd
import pytest

from tslab.analysis.output import AnalysisOutput, FigureFormat, OutputFolder


def small_figure() -> Figure:
    figure = Figure()
    figure.add_subplot().plot([0, 1])
    return figure


def test_tree_layout(tmp_path: Path) -> None:
    output = AnalysisOutput(root=tmp_path / 'out')
    paths = output.save_figure(figure=small_figure(), category='critical_difference', name='cd_all')
    assert paths == (
        tmp_path / 'out/plots/critical_difference/pdf/cd_all.pdf',
        tmp_path / 'out/plots/critical_difference/png/cd_all.png',
    )
    assert all(path.stat().st_size > 0 for path in paths)
    saved = output.save_table(
        frame=pd.DataFrame({'a': [1]}), folder=OutputFolder.STATS, name='wilcoxon'
    )
    assert saved == tmp_path / 'out/stats/wilcoxon.csv'
    assert saved.read_text(encoding='utf-8') == 'a\n1\n'
    assert sorted(path.name for path in (tmp_path / 'out').iterdir()) == ['plots', 'stats']


def test_formats_per_call(tmp_path: Path) -> None:
    paths = AnalysisOutput(root=tmp_path).save_figure(
        figure=small_figure(), category='c', name='n', formats=(FigureFormat.SVG,)
    )
    assert paths == (tmp_path / 'plots/c/svg/n.svg',)


def test_tight_saves_with_label_artists(tmp_path: Path) -> None:
    figure = small_figure()
    figure.axes[0].set_xlabel('a very long label ' * 5)
    paths = AnalysisOutput(root=tmp_path).save_figure(
        figure=figure, category='c', name='n', formats=(FigureFormat.PNG,), tight=True
    )
    assert paths[0].is_file()


def test_unknown_format_string_raises() -> None:
    with pytest.raises(ValueError, match='tiff'):
        FigureFormat('tiff')


def test_empty_formats_and_str_root_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='formats'):
        AnalysisOutput(root=tmp_path, formats=())
    with pytest.raises(TypeError, match='root'):
        AnalysisOutput(root=str(tmp_path))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize('name', ['../x', 'CD Plot', '', '.hidden', 'a/b'])
def test_bad_names_raise(tmp_path: Path, name: str) -> None:
    output = AnalysisOutput(root=tmp_path)
    with pytest.raises(ValueError, match=re.escape(repr(name))):
        output.save_figure(figure=small_figure(), category='c', name=name)
    with pytest.raises(ValueError, match=re.escape(repr(name))):
        output.save_figure(figure=small_figure(), category=name, name='n')
    with pytest.raises(ValueError, match=re.escape(repr(name))):
        output.child(name=name)
    with pytest.raises(ValueError, match=re.escape(repr(name))):
        output.table_path(folder=OutputFolder.TABLES, name=name)
    assert list(tmp_path.iterdir()) == []


def test_child_keeps_settings(tmp_path: Path) -> None:
    output = AnalysisOutput(root=tmp_path, formats=(FigureFormat.EPS,), dpi=100)
    child = output.child(name='per_horizon').child(name='w96_h24')
    assert child == AnalysisOutput(
        root=tmp_path / 'per_horizon/w96_h24', formats=(FigureFormat.EPS,), dpi=100
    )


def test_failed_save_restores_logger_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger('matplotlib.backends.backend_ps')
    logger.setLevel(logging.INFO)
    figure = small_figure()

    def raising(*_args: object, **_kwargs: object) -> None:
        message = 'disk full'
        raise OSError(message)

    monkeypatch.setattr(figure, 'savefig', raising)
    with pytest.raises(OSError, match='disk full'):
        AnalysisOutput(root=tmp_path).save_figure(figure=figure, category='c', name='n')
    assert logger.level == logging.INFO
    assert not any(path.is_file() for path in tmp_path.rglob('*'))
