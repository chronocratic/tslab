from tslab.core.text import describe_error, first_line, format_table, qualified_name


def test_first_line() -> None:
    assert first_line(text='a\nb') == 'a'
    assert first_line(text='') == ''
    assert first_line(text='\nb') == ''


def test_describe_error() -> None:
    assert describe_error(error=ValueError('bad\nvalue')) == 'ValueError: bad\nvalue'


def test_qualified_name() -> None:
    assert qualified_name(cls=ValueError) == 'builtins.ValueError'


def test_format_table_aligns_left_and_strips_trailing_spaces() -> None:
    lines = format_table(header=('a', 'bb'), rows=[('long', 'x')])
    assert lines == ['a     bb', 'long  x']
