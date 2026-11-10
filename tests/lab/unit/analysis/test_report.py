import sys
import types

from matplotlib.figure import Figure
import pandas as pd
import pytest

from tslab.analysis.report import Block, Caption, show


def fake_ipython(*, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    module = types.ModuleType('IPython.display')
    module.HTML = lambda text: ('html', text)  # ty: ignore[unresolved-attribute]
    module.display = calls.append  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, 'IPython.display', module)
    return calls


def test_show_displays_in_order_and_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_ipython(monkeypatch=monkeypatch)
    frame = pd.DataFrame({'a': [1]})
    figure = Figure()
    blocks: list[Block] = [Caption('a < b'), 'x & y', frame, figure]
    show(blocks=blocks)
    assert calls == [
        ('html', '<p><em>a &lt; b</em></p>'),
        ('html', '<pre>x &amp; y</pre>'),
        frame,
        figure,
    ]


def test_caption_plus_text_is_plain_str() -> None:
    assert type(Caption('b') + 'c') is str
    assert isinstance(Caption('[SYNTHETIC DATA] ' + Caption('b')), Caption)


def test_show_rejects_unknown_block(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ipython(monkeypatch=monkeypatch)
    with pytest.raises(TypeError, match='int'):
        show(blocks=[3])  # ty: ignore[invalid-argument-type]
