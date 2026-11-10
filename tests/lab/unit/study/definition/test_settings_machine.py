from tests.lab.unit.study.configs import ClassifyConfig
from tslab.lab.study.definition.settings import non_machine_paths


def test_machine_paths_pass() -> None:
    assert (
        non_machine_paths(config_class=ClassifyConfig, settings={'trainer.accelerator': 'gpu'})
        == []
    )


def test_experiment_path_reported() -> None:
    assert non_machine_paths(
        config_class=ClassifyConfig,
        settings={'trainer.max_epochs': 5, 'trainer.accelerator': 'gpu'},
    ) == ['trainer.max_epochs']


def test_union_needs_every_member_marked() -> None:
    assert non_machine_paths(config_class=ClassifyConfig, settings={'data.num_workers': 2}) == [
        'data.num_workers'
    ]
    assert non_machine_paths(
        config_class=ClassifyConfig, settings={'data': 'ucr', 'data.num_workers': 2}
    ) == ['data']


def test_placeholder_value_on_machine_field() -> None:
    assert (
        non_machine_paths(config_class=ClassifyConfig, settings={'trainer.accelerator': '{cpus}'})
        == []
    )


def test_unknown_path_is_left_to_check_paths() -> None:
    assert non_machine_paths(config_class=ClassifyConfig, settings={'trainer.typo': 1}) == []
