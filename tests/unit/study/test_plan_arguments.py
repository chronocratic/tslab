"""``run_arguments`` and ``validate_arguments``: settings to flags, checked."""

from tests.unit.study.configs import ClassifyConfig, grid
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.plan import run_arguments, validate_arguments

PLACEHOLDER_VALUES = {'output_root': '/x', 'study': 's', 'run_path': 'g/r', 'log_file': '/l'}


def _study() -> Study:
    return Study(name='s', grids={'g': grid()})


def test_run_arguments_expand_then_flag() -> None:
    study = _study()
    run = expand_runs(study=study)[0]
    args = run_arguments(
        grid=study.grids['g'],
        settings={**run.settings, 'data.dataset_folder_path': '${UCR}/Coffee'},
        env={'UCR': '/d'},
        placeholders={
            'output_root': '/x',
            'study': 's',
            'run_path': run.run_path.as_posix(),
            'log_file': '/l',
        },
    )
    assert '/d/Coffee' in args
    assert args[args.index('--output-root') + 1] == '/x/s'
    assert args[args.index('--run-path') + 1] == run.run_path.as_posix()


def test_validate_arguments_accepts_a_good_run() -> None:
    study = _study()
    run = expand_runs(study=study)[0]
    args = run_arguments(
        grid=study.grids['g'], settings=run.settings, env={}, placeholders=PLACEHOLDER_VALUES
    )
    assert validate_arguments(config_class=ClassifyConfig, arguments=args) is None


def test_validate_arguments_returns_tyro_message() -> None:
    message = validate_arguments(config_class=ClassifyConfig, arguments=['model:cost', 'data:ucr'])
    assert message is not None
    assert '--data.dataset-folder-path' in message
    bad = ['model:cost', 'data:ucr', '--data.dataset-folder-path', 'x', '--sed', '1']
    assert validate_arguments(config_class=ClassifyConfig, arguments=bad) is not None
