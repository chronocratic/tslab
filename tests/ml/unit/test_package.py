"""The lazy names of the ml package."""

import importlib

import pytest

import tslab.ml


def test_a_missing_dependency_is_not_hidden_as_a_missing_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = importlib.import_module

    def import_module(name: str, *args: str) -> object:
        if name == 'tslab.ml.interfaces':
            message = "No module named 'lightning'"
            raise ModuleNotFoundError(message, name='lightning')
        return real(name, *args)

    monkeypatch.setattr(tslab.ml, 'import_module', import_module)
    monkeypatch.delitem(vars(tslab.ml), 'Batch', raising=False)
    with pytest.raises(ModuleNotFoundError, match='lightning'):
        tslab.ml.__getattr__('Batch')


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match='no attribute'):
        tslab.ml.not_a_module  # noqa: B018 # why: the access is the test
