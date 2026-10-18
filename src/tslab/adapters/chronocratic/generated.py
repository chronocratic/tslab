"""Shared steps for the generated settings classes of the models and datasets adapters."""

import dataclasses
import sys
from typing import Any


def make_settings_class(
    *, name: str, fields: list[tuple[str, Any, Any]], namespace: dict[str, Any], module: str
) -> type:
    """Create a frozen keyword-only dataclass and publish it as an attribute of ``module``.

    Pickle finds a class by module and name, so the class must be reachable there.
    """
    settings_class = dataclasses.make_dataclass(
        cls_name=name, fields=fields, namespace=namespace, frozen=True, kw_only=True, module=module
    )
    setattr(sys.modules[module], name, settings_class)
    return settings_class


# why: generated classes have no static type
def unknown_settings(*, settings_class: Any, values: dict[str, object]) -> list[str]:  # noqa: ANN401
    """Return the names in ``values`` that are not fields of ``settings_class``, sorted."""
    return sorted(set(values) - {field.name for field in dataclasses.fields(settings_class)})
