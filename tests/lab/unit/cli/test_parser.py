from importlib.util import find_spec
import re

import pytest

from tslab.lab.cli import entrypoint, main, parser

COMMANDS: list[list[str]] = [
    [],
    ['study'],
    ['machine'],
    ['new'],
    ['status'],
    ['study', 'submit'],
    ['study', 'status'],
    ['study', 'list'],
    ['study', 'resubmit'],
    ['study', 'results'],
    ['study', 'logs'],
    ['machine', 'add'],
    ['machine', 'list'],
    ['machine', 'remove'],
]
LEAF_COMMANDS = COMMANDS[3:]


def test_entrypoint_and_group_commands_live_in_sibling_modules() -> None:
    assert find_spec('tslab.lab.cli.entrypoint') is not None
    assert entrypoint.main.__module__ == 'tslab.lab.cli.entrypoint'
    assert vars(parser)['_Study'].__module__ == 'tslab.lab.cli.parser'
    assert vars(parser)['_Machine'].__module__ == 'tslab.lab.cli.parser'


def _command_id(*, command: list[str]) -> str:
    return ' '.join(command) or 'tslab'


def _help(*, command: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as raised:
        main(args=[*command, '--help'])
    assert raised.value.code == 0
    return capsys.readouterr().out


@pytest.mark.parametrize('command', COMMANDS, ids=lambda command: ' '.join(command) or 'tslab')
def test_help_is_plain_and_free_of_flag_pairs(
    *, command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(command=command, capsys=capsys)

    assert '``' not in output
    # why: --no-code-snapshot is a deliberate stand-alone flag, not a bool pair
    assert '--no-' not in output.replace('--no-code-snapshot', '')
    assert '{None}' not in output
    assert '(default: None)' not in output
    assert re.search(r'\b(STR|PATH)$', output.splitlines()[0].strip()) is None


@pytest.mark.parametrize('command', LEAF_COMMANDS, ids=lambda command: ' '.join(command) or 'tslab')
def test_command_help_shows_an_example(
    *, command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(command=command, capsys=capsys)

    assert 'For example:' in output
    assert f'tslab {_command_id(command=command)}' in output


def test_top_help_shows_the_study_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    output = _help(command=[], capsys=capsys)

    assert 'A study, step by step:' in output
    assert 'tslab study resubmit STUDY' in output
    assert 'tslab study wait STUDY --machine local' in output
    assert 'tslab study failures STUDY --machine local' in output


def test_study_group_help_mentions_short_names(capsys: pytest.CaptureFixture[str]) -> None:
    output = _help(command=['study'], capsys=capsys)

    assert 'short name' in output
    assert 'tslab.toml' in output


def test_study_submit_help_names_the_positional(capsys: pytest.CaptureFixture[str]) -> None:
    output = _help(command=['study', 'submit'], capsys=capsys)

    assert 'STUDY' in output


@pytest.mark.parametrize('command', COMMANDS[:3], ids=lambda command: ' '.join(command) or 'tslab')
def test_group_help_lists_one_line_per_command(
    *, command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(command=command, capsys=capsys)

    assert 'For example:' not in output


@pytest.mark.parametrize('command', LEAF_COMMANDS, ids=lambda command: ' '.join(command) or 'tslab')
def test_command_help_has_no_double_blank_lines(
    *, command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(command=command, capsys=capsys)

    assert '\n\n\n' not in output
