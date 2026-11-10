from pathlib import Path

from tslab.lab.study.paths import next_job_file


def test_next_job_file_is_one_past_the_largest_number(tmp_path: Path) -> None:
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    (jobs / '1.sbatch').touch()
    (jobs / '3.tasks').touch()
    (jobs / 'notes.sbatch').touch()
    assert next_job_file(study_folder=tmp_path) == 4


def test_next_job_file_starts_at_one_and_ignores_negative_numbers(tmp_path: Path) -> None:
    assert next_job_file(study_folder=tmp_path) == 1
    (tmp_path / 'jobs').mkdir()
    (tmp_path / 'jobs' / '-3.sbatch').touch()
    assert next_job_file(study_folder=tmp_path) == 1
