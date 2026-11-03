"""What the study commands print: tables, resources, problems, launch results."""

from pathlib import PurePosixPath
from types import MappingProxyType

import pytest

from tslab.core.text import format_table
from tslab.enums import JobPhase, StudyState
from tslab.study.definition import StudyRun
from tslab.study.launchers import LaunchReport, QueuedJob
from tslab.study.launchers.local.backend import LocalResources
from tslab.study.machine import SlurmSettings
from tslab.study.reporting.report import _log_run_lines, log_launch, log_problems, resources_of
from tslab.study.reporting.state import FolderFacts, RunReport


def _queued_report(*, job_id: str) -> RunReport:
    run = StudyRun(
        grid='g',
        values={'small': '1'},
        run_path=PurePosixPath('g/small/1'),
        settings=MappingProxyType({}),
    )
    return RunReport(
        run=run,
        state=StudyState.QUEUED,
        reason=None,
        job_id=job_id,
        job_state='PENDING',
        job_phase=JobPhase.QUEUED,
        attempt=1,
        task=None,
        folder=FolderFacts(snapshot=None, read_error=None, lock_live=False),
    )


def _logged_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_status_line_shows_the_pending_reason(caplog: pytest.LogCaptureFixture) -> None:
    starts = {'1000_0': QueuedJob(start='2026-11-03T10:00:00', reason='Priority', stuck=False)}
    with caplog.at_level('INFO'):
        _log_run_lines(reports=[_queued_report(job_id='1000_0')], starts=starts)
    lines = _logged_lines(caplog)
    assert lines[0].endswith('starts 2026-11-03T10:00:00, waiting: Priority')


def test_status_line_flags_a_stuck_reason(caplog: pytest.LogCaptureFixture) -> None:
    starts = {'1000_0': QueuedJob(start=None, reason='PartitionTimeLimit', stuck=True)}
    with caplog.at_level('INFO'):
        _log_run_lines(reports=[_queued_report(job_id='1000_0')], starts=starts)
    lines = _logged_lines(caplog)
    assert lines[0].endswith('will not start: PartitionTimeLimit')


def test_format_table_gives_every_row_the_same_widths() -> None:
    lines = format_table(
        header=('grid', 'done'), rows=[('a', '1'), ('long-name', '120')], right_align_numbers=True
    )
    assert len({len(line) for line in lines}) == 1
    assert lines[1].startswith('a ')
    assert lines[2].endswith('120')


def test_resources_of_names_every_set_field() -> None:
    assert resources_of(resources=LocalResources(cpus=1, cpus_per_run=1)) == 'local'
    slurm = SlurmSettings(
        partition='gpu', time='01:00:00', qos='high', mem_per_cpu='4G', gpus=2, cpus_per_task=8
    )
    text = resources_of(resources=slurm)
    for part in ('partition=gpu', 'qos=high', 'cpus=8', 'mem/cpu=4G', 'gpus=2'):
        assert part in text
    assert 'qos' not in resources_of(resources=SlurmSettings(partition='gpu', time='01:00:00'))


def test_log_problems_cuts_off_after_twenty(caplog: pytest.LogCaptureFixture) -> None:
    problems = [(PurePosixPath(f'g/{index}'), 'bad') for index in range(25)]
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
