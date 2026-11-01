"""Command-line settings for every chronocratic model, generated from its parameter class.

Every ``X`` in ``chronocratic.models.__all__`` that has a dataclass
``XModelParameters`` gets a frozen settings class ``XSettings``: the same fields
and defaults, minus ``input_dim``, which is taken from the data when the model
is built. ``sequence_length`` is optional; ``None`` takes the length of the data.

Augmentations are given as *recipes*: dataclasses whose ``__call__`` builds an
augmentation producer and is annotated with the view type it produces, for
example ``-> AugmentationProducer[ViewPair]``. A recipe becomes a choice of the
``augmentation`` field of every model that takes that view type, next to
``default``, which keeps the augmentation of the model itself.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import inspect
import operator
import re
import typing
from typing import Annotated, Any

import chronocratic.models
from chronocratic.models.augmentation import AugmentationProducer
import tyro

from tslab.adapters.chronocratic.generated import make_settings_class, unknown_settings

__all__ = [
    'DEFAULT_AUGMENTATION',
    'MODELS',
    'RUNTIME_FIELDS',
    'ChronocraticModelChoice',
    'ModelEntry',
    'chronocratic_model_choice',
    'entry_of',
    'model_settings',
    'recipe_subcommand',
    'recipe_view_type',
]

RUNTIME_FIELDS = frozenset({'input_dim'})
DEFAULT_AUGMENTATION = 'default'
_MODEL_ATTRIBUTE = 'tslab_model'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelEntry:
    """One chronocratic model found in ``chronocratic.models``.

    Attributes:
        name: Subcommand name: the class name in lower case (``'ts2vec'``).
        model_class: The chronocratic ``LightningModule`` class.
        parameters_class: Its ``*ModelParameters`` dataclass.
    """

    name: str
    model_class: Any
    parameters_class: Any


def _discover_models() -> dict[str, ModelEntry]:
    entries: dict[str, ModelEntry] = {}
    for class_name in chronocratic.models.__all__:
        parameters = getattr(chronocratic.models, f'{class_name}ModelParameters', None)
        if isinstance(parameters, type) and dataclasses.is_dataclass(parameters):
            entry = ModelEntry(
                name=class_name.lower(),
                model_class=getattr(chronocratic.models, class_name),
                parameters_class=parameters,
            )
            entries[entry.name] = entry
    return entries


MODELS: Mapping[str, ModelEntry] = _discover_models()


def _view_type(*, annotation: object) -> object | None:
    """Return ``X`` from ``AugmentationProducer[X]``, also inside ``... | None``."""
    for candidate in (annotation, *typing.get_args(annotation)):
        if typing.get_origin(candidate) is AugmentationProducer:
            return typing.get_args(candidate)[0]
    return None


def recipe_view_type(*, recipe: type) -> object:
    """Return the view type the producer of a recipe yields.

    Args:
        recipe: A dataclass whose ``__call__`` is annotated
            ``-> AugmentationProducer[<view type>]``.

    Returns:
        The view type, for example ``ViewPair``.

    Raises:
        TypeError: If ``recipe`` is not a dataclass or does not declare its
            view type.
    """
    if not dataclasses.is_dataclass(recipe):
        message = f'augmentation recipe {recipe.__qualname__} must be a dataclass'
        raise TypeError(message)
    call = inspect.getattr_static(recipe, '__call__', None)
    returns = typing.get_type_hints(call).get('return') if inspect.isfunction(call) else None
    view = _view_type(annotation=returns)
    if view is None:
        message = (
            f'{recipe.__qualname__}.__call__ must be annotated '
            f'-> AugmentationProducer[<view type>], for example '
            f'AugmentationProducer[ViewPair]; got {returns!r}'
        )
        raise TypeError(message)
    return view


def recipe_subcommand(*, recipe: type) -> str:
    """Return the subcommand name: ``JitterThenScaleRecipe`` gives ``jitter-then-scale``."""
    name = recipe.__name__.removesuffix('Recipe')
    # ty types re.findall as list[Any]; cast the well-known shape.
    words = typing.cast('list[str]', re.findall(r'[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+', name))
    return '-'.join(words).lower()


def _check_recipes(*, recipes: Sequence[type]) -> None:
    names = [recipe_subcommand(recipe=recipe) for recipe in recipes]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        message = f'augmentation recipe names {repeated} appear more than once'
        raise ValueError(message)
    if DEFAULT_AUGMENTATION in names:
        message = (
            f'a recipe may not be named {DEFAULT_AUGMENTATION!r}; that name keeps the model default'
        )
        raise ValueError(message)
    model_views = [
        _view_type(annotation=_hints(entry=entry).get('augmentation')) for entry in MODELS.values()
    ]
    for recipe in recipes:
        if recipe_view_type(recipe=recipe) not in model_views:
            message = (
                f'augmentation recipe {recipe.__qualname__} produces '
                f'{recipe_view_type(recipe=recipe)!r}, which fits no model'
            )
            raise TypeError(message)


def _hints(*, entry: ModelEntry) -> dict[str, Any]:
    return typing.get_type_hints(entry.parameters_class, include_extras=True)


def _augmentation_type(*, annotation: object, recipes: Sequence[Any]) -> object | None:
    view = _view_type(annotation=annotation)
    if view is None:
        return None
    matching = [recipe for recipe in recipes if recipe_view_type(recipe=recipe) == view]
    if not matching:
        return None
    members: list[Any] = [
        Annotated[recipe, tyro.conf.subcommand(name=recipe_subcommand(recipe=recipe))]
        for recipe in matching
    ]
    members.append(Annotated[None, tyro.conf.subcommand(name=DEFAULT_AUGMENTATION)])
    return functools.reduce(operator.or_, members)


# why: a dataclasses.field() placeholder is typed Any by the standard library
def _copied_default(*, field: dataclasses.Field[Any]) -> Any:  # noqa: ANN401
    if field.default_factory is not dataclasses.MISSING:
        return dataclasses.field(default_factory=field.default_factory)
    if field.default is not dataclasses.MISSING:
        return dataclasses.field(default=field.default)
    return dataclasses.field()


def _model_field(
    *, field: dataclasses.Field[Any], hints: Mapping[str, Any], recipes: Sequence[type]
) -> tuple[str, Any, Any] | None:
    """The settings field of one parameter of a model, or ``None`` when it is left out."""
    if field.name in RUNTIME_FIELDS:
        return None
    if field.name == 'augmentation':
        augmentation = _augmentation_type(annotation=hints['augmentation'], recipes=recipes)
        return (
            None
            if augmentation is None
            else ('augmentation', augmentation, dataclasses.field(default=None))
        )
    if field.name == 'sequence_length':
        return 'sequence_length', int | None, dataclasses.field(default=None)
    return field.name, hints[field.name], _copied_default(field=field)


def _check_recipe(*, name: str, recipe: object) -> None:
    """Raise unless the augmentation recipe of the settings produces the view the model takes."""
    if recipe is None:
        return
    model_view = _view_type(annotation=_hints(entry=MODELS[name]).get('augmentation'))
    if recipe_view_type(recipe=type(recipe)) != model_view:
        message = (
            f'{type(recipe).__qualname__} does not fit model {name!r}, which takes '
            f'{model_view!r}; choose a recipe that produces it'
        )
        raise TypeError(message)


def _settings_class(*, entry: ModelEntry, recipes: Sequence[type]) -> type:
    hints = _hints(entry=entry)
    maybe_fields = (
        _model_field(field=field, hints=hints, recipes=recipes)
        for field in dataclasses.fields(entry.parameters_class)
    )
    fields = [item for item in maybe_fields if item is not None]
    # Another set of recipes gives other classes, so each set gets its own names.
    suffix = ''.join(f'_{recipe.__name__}' for recipe in recipes)
    return make_settings_class(
        name=f'{entry.model_class.__name__}Settings{suffix}',
        fields=fields,
        namespace={_MODEL_ATTRIBUTE: entry.name, '__doc__': entry.parameters_class.__doc__},
        module=__name__,
    )


# why: the union is built at run time; Any lets it serve as a field annotation
def chronocratic_model_choice(*, augmentations: Sequence[type] = ()) -> Any:  # noqa: ANN401
    """Return the command-line choice between all chronocratic models.

    Use the result as a config field annotation, ``model: ModelChoice``; on the
    command line, ``model:ts2vec`` selects a model and ``--model.<field>``
    sets its settings. The same recipes always give the same union.

    Args:
        augmentations: Augmentation recipes. Each appears as
            ``model.augmentation:<name>`` on every model whose augmentation
            takes its view type.

    Returns:
        A union of the generated settings classes.

    Raises:
        TypeError: If a recipe is not a dataclass, does not declare its view
            type, or fits no model.
        ValueError: If two recipes have the same name, or one is named
            ``default``.
    """
    return _model_choice(augmentations=tuple(augmentations))


# Pickle finds a settings class by module and name, so the classes of one set of
# recipes are generated once; a second call returns the same union.
@functools.cache
def _model_choice(*, augmentations: tuple[type, ...]) -> Any:  # noqa: ANN401 # why: run-time union
    _check_recipes(recipes=augmentations)
    members: list[Any] = []
    for name, entry in MODELS.items():
        settings_class: Any = _settings_class(entry=entry, recipes=augmentations)
        members.append(Annotated[settings_class, tyro.conf.subcommand(name=name)])
    return functools.reduce(operator.or_, members)


ChronocraticModelChoice = chronocratic_model_choice()


def entry_of(*, settings: object) -> ModelEntry:
    """Return the model a settings object belongs to.

    Raises:
        TypeError: If ``settings`` was not made by ``chronocratic_model_choice``.
    """
    name = getattr(type(settings), _MODEL_ATTRIBUTE, None)
    if not isinstance(name, str) or name not in MODELS:
        message = (
            f'{type(settings).__qualname__} is not chronocratic model settings; '
            'use a model chosen through chronocratic_model_choice'
        )
        raise TypeError(message)
    return MODELS[name]


# why: the settings classes are generated at run time
def model_settings(*, choice: Any, name: str, **values: object) -> Any:  # noqa: ANN401
    """Return the settings of one model, for writing presets in code.

    The settings classes are generated, so they cannot be imported; this
    function finds the one behind a subcommand name in ``choice``.

    Args:
        choice: A union returned by ``chronocratic_model_choice``.
        name: The subcommand name of the model, for example ``'ts2vec'``.
        **values: Settings that differ from the defaults.

    Raises:
        TypeError: If ``choice`` was not made by ``chronocratic_model_choice``, or the
            augmentation recipe produces a view type the model does not take.
        ValueError: If no model has this name, or the model has no such setting.
    """
    classes: dict[str, Any] = {}
    for member in typing.get_args(choice):
        settings_class = typing.get_args(member)[0] if typing.get_args(member) else member
        model_name = getattr(settings_class, _MODEL_ATTRIBUTE, None)
        if not isinstance(model_name, str):
            message = 'choice must be a union returned by chronocratic_model_choice'
            raise TypeError(message)
        classes[model_name] = settings_class
    if name not in classes:
        message = f'no model {name!r}; choose from {sorted(classes)}'
        raise ValueError(message)
    unknown = unknown_settings(settings_class=classes[name], values=values)
    if unknown:
        message = f'model {name!r} has no settings {unknown}'
        raise ValueError(message)
    _check_recipe(name=name, recipe=values.get('augmentation'))
    return classes[name](**values)
