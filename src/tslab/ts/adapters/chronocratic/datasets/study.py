"""Dataset axes for the study layer, over the chronocratic archives.

A user can write the same mapping by hand in a study module; these helpers
save the spelling of the generated fields. Both archives accept the same
fields, so one axis can hold both, ``ucr_datasets(...) | uea_datasets(...)``.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from tslab.core.paths import require_path
from tslab.study.definition import SettingValue

__all__ = ['ucr_datasets', 'uea_datasets']


def ucr_datasets(
    *,
    root: Path,
    names: Sequence[str],
    target_column_name: str = 'target',
    settings: Mapping[str, SettingValue] | None = None,
) -> dict[str, dict[str, SettingValue]]:
    """Axis values that point at the UCR archives under ``root``.

    Args:
        root: The folder that holds the dataset folders, e.g. ``Path('${UCR_ROOT}')``.
        names: The UCR dataset names, e.g. ``('Coffee',)``.
        target_column_name: The UCR column name the model trains on.
        settings: Extra settings every dataset gets, e.g. ``{'data.batch_size': 2}``.

    Returns:
        ``{name: {'data': 'ucr', 'data.dataset_folder_path': ..., ...}}``, one
        entry per name, with the extra settings last.

    Raises:
        TypeError: If ``root`` is a string.
        ValueError: If ``names`` is empty.
    """
    return _datasets(
        archive='ucr',
        root=root,
        names=names,
        target_column_name=target_column_name,
        settings=settings,
    )


def uea_datasets(
    *,
    root: Path,
    names: Sequence[str],
    target_column_name: str = 'target',
    settings: Mapping[str, SettingValue] | None = None,
) -> dict[str, dict[str, SettingValue]]:
    """Axis values that point at the UEA archives under ``root``.

    Args:
        root: The folder that holds the dataset folders, e.g. ``Path('/d/UEA')``.
        names: The UEA dataset names, e.g. ``('BasicMotions',)``.
        target_column_name: The UEA column name the model trains on.
        settings: Extra settings every dataset gets, e.g. ``{'data.batch_size': 2}``.

    Returns:
        ``{name: {'data': 'uea', 'data.dataset_folder_path': ..., ...}}``, one
        entry per name, with the extra settings last.

    Raises:
        TypeError: If ``root`` is a string.
        ValueError: If ``names`` is empty.
    """
    return _datasets(
        archive='uea',
        root=root,
        names=names,
        target_column_name=target_column_name,
        settings=settings,
    )


def _datasets(
    *,
    archive: str,
    root: Path,
    names: Sequence[str],
    target_column_name: str,
    settings: Mapping[str, SettingValue] | None,
) -> dict[str, dict[str, SettingValue]]:
    require_path(value=root, parameter='root')
    if not names:
        message = f'{archive}_datasets needs at least one dataset name'
        raise ValueError(message)
    extra = dict(settings) if settings is not None else {}
    return {
        name: {
            'data': archive,
            'data.dataset_folder_path': (root / name).as_posix(),
            'data.target_column_name': target_column_name,
            **extra,
        }
        for name in names
    }
