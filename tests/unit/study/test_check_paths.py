"""``check_paths``: which setting paths exist on no reading of the config."""

from study.configs import ClassifyConfig
from tslab.study import check_paths


def test_paths_accepted_in_any_member_when_unchosen() -> None:
    settings = {'model.alpha': 0.1, 'model.depth': 3}
    assert check_paths(config_class=ClassifyConfig, settings=settings) == []


def test_chosen_member_is_strict() -> None:
    settings = {'model': 'cost', 'model.depth': 3}
    assert check_paths(config_class=ClassifyConfig, settings=settings) == ['model.depth']


def test_typos_and_bad_choices_are_reported() -> None:
    settings = {'sed': 1, 'model': 'tsvec', 'trainer.max_epoch': 3}
    assert check_paths(config_class=ClassifyConfig, settings=settings) == [
        'sed',
        'model',
        'trainer.max_epoch',
    ]
