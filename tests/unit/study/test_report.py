"""What the study commands print: tables, resources, problems, launch results."""

import pytest

from tslab.study.launch import LaunchReport
from tslab.study.machine import SlurmSettings
from tslab.study.report import format_table, log_launch, log_problems, resources_of


def test_format_table_gives_every_row_the_same_widths() -> None:
    lines = format_table(header=('grid', 'done'), rows=[('a', '1'), ('long-name', '120')])
    assert len({len(line) for line in lines}) == 1
    assert lines[1].startswith('a ')
    assert lines[2].endswith('120')


def test_resources_of_names_every_set_field() -> None:
    assert resources_of(slurm=None) == 'local'
    slurm = SlurmSettings(
        partition='gpu', time='01:00:00', qos='high', mem_per_cpu='4G', gpus=2, cpus_per_task=8
    )
    text = resources_of(slurm=slurm)
    for part in ('partition=gpu', 'qos=high', 'cpus=8', 'mem/cpu=4G', 'gpus=2'):
        assert part in text
    assert 'qos' not in resources_of(slurm=SlurmSettings(partition='gpu', time='01:00:00'))


def test_log_problems_cuts_off_after_twenty(caplog: pytest.LogCaptureFixture) -> None:
    problems = [(f'g/{index}', 'bad') for index in range(25)]
    with caplog.at_level('ERROR'):
        log_problems(problems=problems)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 21
    assert messages[-1].endswith('5 more')


def test_log_launch_reports_a_submission_error(caplog: pytest.LogCaptureFixture) -> None:
    report = LaunchReport(submitted=(), not_submitted=(), error='sbatch exploded')
    with caplog.at_level('INFO'):
        code, submitted = log_launch(report=report, groups=[])
    assert (code, submitted) == (1, [])
    assert 'sbatch exploded' in caplog.text
