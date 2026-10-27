from tslab.core.text import describe_error, qualified_name


def test_describe_error() -> None:
    assert describe_error(error=ValueError('bad\nvalue')) == 'ValueError: bad\nvalue'


def test_qualified_name() -> None:
    assert qualified_name(cls=ValueError) == 'builtins.ValueError'
