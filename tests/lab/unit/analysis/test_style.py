import matplotlib as mpl
from matplotlib.figure import Figure

from tslab.lab.analysis.style import is_dark_cell, paper_style


def test_paper_style_sets_and_restores() -> None:
    before = mpl.rcParams['axes.labelsize']
    with paper_style(font_size=7):
        assert mpl.rcParams['axes.labelsize'] == 7
        assert mpl.rcParams['figure.dpi'] == 150
    assert mpl.rcParams['axes.labelsize'] == before


def test_is_dark_cell_on_grey_ends() -> None:
    image = Figure().add_subplot().imshow([[0.0, 1.0]], cmap='gray', vmin=0, vmax=1)
    assert is_dark_cell(image=image, value=0.0)
    assert not is_dark_cell(image=image, value=1.0)
