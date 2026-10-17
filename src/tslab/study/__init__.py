"""TSLab studies: one saved definition of a round of experiments.

A study turns every run into the flags of the user's own ``no-preset``
command and hands them to a launcher; TSLab never runs experiment code.
"""

from tslab.study.settings import from_presets, Settings, settings_to_args

__all__ = ['Settings', 'from_presets', 'settings_to_args']
