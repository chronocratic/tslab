import math
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from tslab.analysis.cd_diagram import cd_diagram, cd_groups, critical_difference
from tslab.analysis.stats import Direction


@pytest.mark.parametrize(('k', 'q'), [(2, 1.960), (5, 2.728), (10, 3.164)])
def test_q_alpha_matches_table(k: int, q: float) -> None:
    n = 10
    cd = critical_difference(method_count=k, dataset_count=n)
    assert round(cd / math.sqrt(k * (k + 1) / (6 * n)), 3) == q


def test_cd_groups_drop_contained() -> None:
    assert cd_groups(ranks=[1.0, 1.2, 1.4, 3.0, 3.5], cd=1.0) == [(0, 2), (3, 4)]
    assert cd_groups(ranks=[1.0, 1.5, 2.4, 3.0, 3.9], cd=1.0) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_cd_groups_equal_ranks() -> None:
    assert cd_groups(ranks=[2.0, 2.0, 2.0], cd=0.5) == [(0, 2)]


def clear_winner(rng: np.random.Generator) -> pd.DataFrame:
    base = rng.normal(size=(12, 1))
    offsets = np.array([[3.0, 2.0, 1.0, 0.0]])
    scores = base + offsets + rng.normal(scale=0.1, size=(12, 4))
    return pd.DataFrame(scores, columns=['best', 'second', 'third', 'last'])


def test_cd_diagram_draws_one_label_per_method() -> None:
    result = cd_diagram(
        matrix=clear_winner(np.random.default_rng(0)),
        direction=Direction.HIGHER_IS_BETTER,
        labels={'best': 'Best method'},
    )
    assert result.figure is not None
    texts = [text.get_text() for axis in result.figure.axes for text in axis.texts]
    assert sum('Best method' in text for text in texts) == 1
    assert sum(text == 'last' for text in texts) == 1
    assert result.ranks.index[0] == 'best'
    assert result.caption.startswith('Critical-difference diagram over 12 datasets and 4 methods')


def test_cd_diagram_none_when_not_significant() -> None:
    # every method takes every rank twice: equal rank sums, Friedman p = 1
    rows = [[1, 2, 3], [2, 3, 1], [3, 1, 2]] * 2
    matrix = pd.DataFrame(rows, columns=list('abc'), dtype=float)
    result = cd_diagram(matrix=matrix, direction=Direction.HIGHER_IS_BETTER)
    assert result.figure is None
    assert 'finds no difference' in result.caption


def test_cd_diagram_none_when_too_few() -> None:
    matrix = pd.DataFrame({'a': [1.0, 2.0], 'b': [2.0, 3.0]})
    result = cd_diagram(matrix=matrix, direction=Direction.HIGHER_IS_BETTER)
    assert result.figure is None
    assert result.caption.startswith('No critical-difference diagram: 2 complete datasets')


def test_cd_diagram_drops_incomplete_rows() -> None:
    matrix = clear_winner(np.random.default_rng(0))
    matrix.iloc[0, 1] = np.nan
    result = cd_diagram(matrix=matrix, direction=Direction.HIGHER_IS_BETTER)
    assert result.friedman.datasets == 11
    assert 'over 11 datasets' in result.caption


def test_cd_diagram_uses_no_pyplot(tmp_path: Path) -> None:
    script = tmp_path / 'check.py'
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import numpy as np
            import pandas as pd
            from tslab.analysis.cd_diagram import cd_diagram
            from tslab.analysis.stats import Direction

            rng = np.random.default_rng(0)
            matrix = pd.DataFrame(
                rng.normal(size=(12, 4)) + np.array([[3.0, 2.0, 1.0, 0.0]]), columns=list('abcd')
            )
            assert cd_diagram(matrix=matrix, direction=Direction.HIGHER_IS_BETTER).figure
            assert 'matplotlib.pyplot' not in sys.modules
            """
        ),
        encoding='utf-8',
    )
    subprocess.run([sys.executable, str(script)], check=True)  # noqa: S603 # why: runs this interpreter on a fixed script
