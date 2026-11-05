"""Preset axes without experiment settings owned by each study."""

from tslab.study import from_presets, SettingValue

from {{package_name}} import classify, forecast

__all__ = ['classification_recipes', 'forecasting_recipes']

_TRAINING_SETTINGS = (
    'trainer.max_epochs',
    'trainer.max_training_time',
    'trainer.early_stopping_patience',
    'trainer.early_stopping_min_delta',
)


def classification_recipes() -> dict[str, dict[str, SettingValue]]:
    """Return classification presets without settings owned by each study."""
    return from_presets(
        presets=classify.PRESETS,
        names=('ts2vec-archive-split',),
        without=('data', 'run_path', *_TRAINING_SETTINGS),
    )


def forecasting_recipes() -> dict[str, dict[str, SettingValue]]:
    """Return forecasting presets without settings owned by each study."""
    return from_presets(
        presets=forecast.PRESETS,
        names=('ts2vec-full-series', 'cost-cropped'),
        without=('data', 'run_path', *_TRAINING_SETTINGS),
    )
