"""Shell text for local job scripts."""

from collections.abc import Mapping
import re
import shlex

__all__ = ['export_lines']

_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def export_lines(*, env: Mapping[str, str]) -> list[str]:
    """Format environment variables as quoted shell exports.

    Args:
        env: Environment names and values, in output order.

    Returns:
        One shell export line per environment variable.

    Raises:
        ValueError: If a name is not a shell variable name.
    """
    lines = []
    for name, value in env.items():
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            message = f'{name!r} is not a valid environment variable name'
            raise ValueError(message)
        lines.append(f'export {name}={shlex.quote(value)}')
    return lines
