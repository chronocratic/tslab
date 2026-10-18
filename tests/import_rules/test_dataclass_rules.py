"""Every dataclass in the repository is frozen and keyword-only."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DECORATOR = re.compile(r'^\s*@(?:dataclasses\.)?dataclass\b(.*)$')


def dataclass_decorators() -> list[tuple[str, int, str]]:
    # Line-based, not ast: template files hold {{placeholders}} and do not parse.
    files = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '*.py', '*.md'],  # noqa: S607 # why: fixed git command, no user input
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found = []
    for name in files:
        if name.startswith('planning/') or name == 'CHANGELOG.md':
            continue
        if not (ROOT / name).is_file():
            continue  # listed by git but deleted in the working tree
        lines = (ROOT / name).read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            match = DECORATOR.match(line)
            if match:
                arguments = str(match.group(1))
                # A decorator may span several lines; read on until the parenthesis closes.
                following = iter(lines[number:])
                while arguments.count('(') > arguments.count(')'):
                    arguments += next(following, ')')
                found.append((name, number, arguments))
    return found


def test_every_dataclass_is_frozen_and_keyword_only() -> None:
    decorators = dataclass_decorators()
    assert decorators, 'the scan found no dataclasses; the pattern or file list is wrong'
    bad = [
        f'{name}:{number}'
        for name, number, arguments in decorators
        if 'frozen=True' not in arguments or 'kw_only=True' not in arguments
    ]
    assert bad == []
