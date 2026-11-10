"""SLURM script-building helpers."""

from pathlib import Path

import pytest

from tslab.lab.jobs.slurm import (
    array_spec,
    check_script_text,
    export_lines,
    node_local_tmp_lines,
    sbatch_directives,
    srun_command,
    wait_for_file_line,
)


def test_array_spec_groups_consecutive_indices() -> None:
    assert array_spec(indices=(10, 9, 0, 1, 2, 3, 7, 3)) == '0-3,7,9-10'
    assert array_spec(indices=(4,)) == '4'
    with pytest.raises(ValueError, match='indices'):
        array_spec(indices=())


def test_sbatch_directives_preserve_order_quote_values_and_skip_false() -> None:
    assert sbatch_directives(
        options={
            'job_name': 'my job',
            'exclusive': True,
            'quiet': False,
            'output': '/project/a b/%a.out',
        }
    ) == [
        "#SBATCH --job-name='my job'",
        '#SBATCH --exclusive',
        "#SBATCH --output='/project/a b/%a.out'",
    ]


@pytest.mark.parametrize('key', ['1nodes', 'bad key', 'node;touch'])
def test_sbatch_directives_reject_invalid_option_names(key: str) -> None:
    with pytest.raises(ValueError, match='option name'):
        sbatch_directives(options={key: 1})


def test_sbatch_directives_reject_control_characters() -> None:
    with pytest.raises(ValueError, match='control character'):
        sbatch_directives(options={'comment': 'safe\nunsafe'})


def test_check_script_text_rejects_control_characters() -> None:
    with pytest.raises(ValueError, match='control character'):
        check_script_text(value='first\nsecond', what='value')


def test_export_lines_quote_values() -> None:
    assert export_lines(env={'EXAMPLE': 'a b', 'EMPTY': ''}) == [
        "export EXAMPLE='a b'",
        "export EMPTY=''",
    ]


def test_export_lines_preserve_quoted_multiline_values() -> None:
    assert export_lines(env={'EXAMPLE': 'first\nsecond'}) == ["export EXAMPLE='first\nsecond'"]


def test_export_lines_reject_invalid_names() -> None:
    with pytest.raises(ValueError, match='environment variable name'):
        export_lines(env={'BAD-NAME': 'value'})


def test_node_local_tmp_lines_use_prefix_and_array_id() -> None:
    assert node_local_tmp_lines(prefix='worker', array=True) == [
        'TMPDIR=/tmp/worker-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}',
        'mkdir -p "$TMPDIR"',
        'trap \'rm -rf "$TMPDIR"\' EXIT',
        'export TMPDIR',
    ]


def test_wait_for_file_line_quotes_path_and_uses_timeout() -> None:
    assert wait_for_file_line(path=Path('/project/jobs with space/0.tasks'), seconds=7) == (
        "for _ in $(seq 7); do [ -s '/project/jobs with space/0.tasks' ] && break; sleep 1; done"
    )


def test_wait_for_file_line_rejects_string_paths() -> None:
    with pytest.raises(TypeError, match='path must be a Path'):
        wait_for_file_line(path='/project/jobs/0.tasks', seconds=7)  # ty: ignore[invalid-argument-type]


def test_srun_command_runs_variable_with_expected_flags() -> None:
    assert srun_command(variable='TASK_LINE') == (
        'srun --unbuffered --kill-on-bad-exit=1 bash -c "exec $TASK_LINE"'
    )
