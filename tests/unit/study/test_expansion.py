"""``expand_variables``, ``expand_placeholders`` and ``expand_setting``."""

from pathlib import Path

import pytest

from tslab.study.expansion import expand_placeholders, expand_setting, expand_variables


def test_variables() -> None:
    assert expand_variables(text='${UCR_ROOT}/Coffee', env={'UCR_ROOT': '/d'}) == '/d/Coffee'
    assert expand_variables(text='cost $5', env={}) == 'cost $5'


def test_unset_variable_raises() -> None:
    with pytest.raises(ValueError, match='UCR_ROOT'):
        expand_variables(text='${UCR_ROOT}/Coffee', env={})


def test_placeholders() -> None:
    values = {
        'output_root': '/x',
        'study': 'example',
        'run_path': 'g/a/b',
        'log_file': '/x/example/.study/logs/cls_5.out',
    }
    assert expand_placeholders(text='{output_root}/{study}', values=values) == '/x/example'
    with pytest.raises(ValueError, match='seed'):
        expand_placeholders(text='{seed}', values=values)


def test_expand_setting_types() -> None:
    env, values = {'R': '/d'}, {'study': 's'}
    assert expand_setting(value=Path('${R}/{study}'), env=env, values=values) == Path('/d/s')
    assert expand_setting(value=('${R}', 3), env=env, values=values) == ('/d', 3)
    assert expand_setting(value=7, env=env, values=values) == 7


def test_env_values_are_not_scanned_for_placeholders() -> None:
    values = {'study': 's'}
    assert expand_setting(value='${R}/{study}', env={'R': '{study}'}, values=values) == '{study}/s'
    with pytest.raises(ValueError, match='nope'):
        expand_setting(value='{nope}', env={}, values=values)


def test_doubled_braces_escape_a_literal_brace() -> None:
    values = {'study': 's'}
    assert expand_placeholders(text='{{epoch}}-{study}', values=values) == '{epoch}-s'
    assert expand_setting(value='ckpt-{{epoch}}', env={}, values=values) == 'ckpt-{epoch}'
