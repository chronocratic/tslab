"""Markers for settings controlled by a machine rather than a study."""

from typing import Annotated, get_args, get_origin

__all__ = ['HardwareSetting', 'MachineSetting', 'is_hardware_setting', 'is_machine_setting']


class MachineSetting:
    """May be set in a machine profile: what the machine decides, never the experiment."""


class HardwareSetting(MachineSetting):
    """A machine setting that names the hardware the run uses."""


def _matches_marker(*, metadata: object, marker: type[MachineSetting]) -> bool:
    return (isinstance(metadata, type) and issubclass(metadata, marker)) or isinstance(
        metadata, marker
    )


def _has_marker(*, annotation: object, marker: type[MachineSetting]) -> bool:
    if get_origin(annotation) is not Annotated:
        return False
    return any(
        _matches_marker(metadata=metadata, marker=marker) for metadata in get_args(annotation)[1:]
    )


def is_machine_setting(*, annotation: object) -> bool:
    """Return whether an ``Annotated`` field may be set in a machine profile.

    Args:
        annotation: A type annotation, possibly ``Annotated``.

    Returns:
        Whether the annotation carries ``MachineSetting`` or one of its subclasses.
    """
    return _has_marker(annotation=annotation, marker=MachineSetting)


def is_hardware_setting(*, annotation: object) -> bool:
    """Return whether an ``Annotated`` field names hardware used by the run.

    Args:
        annotation: A type annotation, possibly ``Annotated``.

    Returns:
        Whether the annotation carries ``HardwareSetting``.
    """
    return _has_marker(annotation=annotation, marker=HardwareSetting)
