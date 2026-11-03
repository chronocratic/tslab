"""Saving, loading and parsing run configurations.

Configs are plain frozen dataclasses. This module writes them as plain YAML
and reads them back. The strict, typed conversion between the dataclass and
plain data lives in ``tslab.core.codec``: decoding only ever constructs types
reachable from the annotations of the config class (safe, unlike pickle or
the full loader of PyYAML), and unknown keys, wrong value types and unknown
union members raise ``ConfigFileError`` naming the exact field, so a typo in
a hand-edited file is never silently ignored.

Parsing the command line is delegated to tyro; ``parse_config`` only adds
the defaults of TSLab.

``REQUIRED`` marks a preset value its author cannot know (a dataset path, a
cluster folder). A preset that carries it must be given the value on the
command line, or tyro stops with ``Missing ... --<flag>``. A parsed config
never holds the marker, and ``save_config`` raises ``TypeError`` for an
unparsed preset that still holds it.
"""

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
from typing import Annotated, Any, Union

import tyro
import yaml

from tslab.core.files import atomic_write_text
from tslab.core.paths import require_path
from tslab.core.text import qualified_name

__all__ = [
    'CONFIG_FORMAT_VERSION',
    'NO_PRESET',
    'REQUIRED',
    'ConfigFileError',
    'config_from_yaml',
    'config_to_yaml',
    'load_config',
    'parse_config',
    'save_config',
]

_LOGGER = logging.getLogger(__name__)
CONFIG_FORMAT_VERSION = 1

# An optional marker for a preset value the preset author cannot know; the
# user must give it on the command line (tyro stops with "Missing ...").
REQUIRED: Any = tyro.MISSING  # why: tyro.MISSING has no public type
NO_PRESET = 'no-preset'


class ConfigFileError(ValueError):
    """Raised when a config file does not match the config class."""


def config_to_yaml(*, config: object) -> str:
    """Serialize a config dataclass to YAML.

    Args:
        config: A dataclass instance.

    Returns:
        YAML text with ``format_version``, ``config_type`` and ``config`` keys.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or holds a value
            that config files cannot represent.
    """
    # why: core.codec imports ConfigFileError from this module at module level,
    # so a module-level import back would cycle
    from tslab.core.codec import config_to_data  # noqa: PLC0415

    document = {
        'format_version': CONFIG_FORMAT_VERSION,
        'config_type': qualified_name(cls=type(config)),
        'config': config_to_data(config=config),
    }
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def config_from_yaml[C](*, config_class: type[C], text: str) -> C:
    """Rebuild a config from YAML written by ``config_to_yaml``.

    Args:
        config_class: The dataclass to build.
        text: YAML text.

    Returns:
        The config instance.

    Raises:
        ConfigFileError: If the text does not match ``config_class``.
    """
    # why: core.codec imports ConfigFileError from this module at module level,
    # so a module-level import back would cycle
    from tslab.core.codec import decode_dataclass  # noqa: PLC0415

    document = yaml.safe_load(text)
    if not isinstance(document, dict) or document.get('format_version') != CONFIG_FORMAT_VERSION:
        message = f'not a TSLab config file (format_version {CONFIG_FORMAT_VERSION} expected)'
        raise ConfigFileError(message)
    if document.get('config_type') != qualified_name(cls=config_class):
        _LOGGER.warning(
            'config file was written for %s, loading it as %s',
            document.get('config_type'),
            qualified_name(cls=config_class),
        )
    config = decode_dataclass(data=document.get('config'), cls=config_class, location='config')
    if not isinstance(config, config_class):
        message = f'decoded config is {type(config).__qualname__}, not {config_class.__qualname__}'
        raise ConfigFileError(message)
    return config


def save_config(*, config: object, path: Path) -> None:
    """Write ``config`` to ``path`` atomically (see ``config_to_yaml``).

    Args:
        config: A dataclass instance.
        path: Destination, usually ``<run folder>/config.yaml``.
    """
    require_path(value=path, parameter='path')
    atomic_write_text(path=path, text=config_to_yaml(config=config))


def load_config[C](*, config_class: type[C], path: Path) -> C:
    """Load a config saved by a run, for example to reproduce it.

    Example::

        config = tslab.load_config(config_class=ExperimentConfig, path=Path('runs/a/config.yaml'))
        config = tslab.parse_config(
            config_class=ExperimentConfig, default=config, args=['--seed', '3']
        )

    Args:
        config_class: The dataclass to build.
        path: A ``config.yaml`` written by ``Run``.

    Returns:
        The config instance.
    """
    require_path(value=path, parameter='path')
    return config_from_yaml(config_class=config_class, text=path.read_text(encoding='utf-8'))


def parse_config[C](
    *,
    config_class: type[C],
    presets: Mapping[str, tuple[str, C]] | None = None,
    default: C | None = None,
    args: Sequence[str] | None = None,
) -> C:
    """Build a config from the command line with tyro.

    Flags may appear before or after union subcommands such as ``model:ts2vec``
    (``CascadeSubcommandArgs`` of tyro), so ``model:ts2vec --seed 3`` and
    ``--seed 3 model:ts2vec`` both work.

    Args:
        config_class: The config dataclass.
        presets: Named presets ``{name: (description, config)}``, the
            experiment registry. The first argument then selects a preset or
            ``no-preset``. The union choices of a preset (its model, its dataset)
            are fixed, while flags still override its values; ``no-preset``
            starts from ``config_class`` with every choice open.
        default: Starting values, for example a config loaded with ``load_config``.
            Not allowed together with ``presets``.
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed config.

    Raises:
        ValueError: If both ``presets`` and ``default`` are given, or a preset
            is named ``no-preset``.
    """
    cascade = (tyro.conf.CascadeSubcommandArgs, tyro.conf.EnumChoicesFromValues)
    if presets is not None:
        return _parse_preset(
            config_class=config_class, presets=presets, default=default, args=args, config=cascade
        )
    if default is None:
        return tyro.cli(config_class, args=args, config=cascade)
    return tyro.cli(config_class, default=default, args=args, config=cascade)


def _parse_preset[C](
    *,
    config_class: type[C],
    presets: 'Mapping[str, tuple[str, C]]',
    default: C | None,
    args: Sequence[str] | None,
    config: tuple[object, ...],
) -> C:
    """Parse the arguments with the first argument selecting a preset."""
    if default is not None:
        message = 'pass either presets= or default=, not both'
        raise ValueError(message)
    choice = _preset_choice(config_class=config_class, presets=presets)
    parsed = tyro.cli(choice, args=args, config=config)
    # tyro returns Any; every preset is an instance of config_class.
    if not isinstance(parsed, config_class):
        message = f'a preset is not a {config_class.__qualname__}: {type(parsed).__qualname__}'
        raise TypeError(message)
    return parsed


type _Presets[C] = Mapping[str, tuple[str, C]]


# why: the return value is a union type built at run time, which no annotation can name
def _preset_choice[C](*, config_class: type[C], presets: _Presets[C]) -> Any:  # noqa: ANN401
    """Return the tyro union of ``no-preset`` and the presets.

    ``AvoidSubcommands`` on a preset turns each union field that has a default
    into the type of that default, so ``model:other`` is rejected instead of
    silently dropping the model settings of the preset.
    """
    if NO_PRESET in presets:
        message = f'{NO_PRESET!r} is reserved for starting without a preset'
        raise ValueError(message)
    free: Any = config_class  # why: ty accepts a class in Annotated only when typed Any
    members: list[Any] = [  # why: the members are Annotated types built at run time
        Annotated[
            free, tyro.conf.subcommand(name=NO_PRESET, description='Every setting from the flags.')
        ]
    ]
    for name, (description, config) in presets.items():
        preset_class: Any = type(config)  # why: a preset's class is only known at run time
        members.append(
            Annotated[
                preset_class,
                tyro.conf.subcommand(name=name, default=config, description=description),
                tyro.conf.AvoidSubcommands,
            ]
        )
    return Union[tuple(members)]  # noqa: UP007 # why: built from a runtime tuple
