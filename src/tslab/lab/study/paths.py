"""The folder paths of a study: the study folder and the files it holds.

These are pure path computations shared by planning, the launchers, the
commands and the result table; they take the already-resolved values and
return a ``Path`` or an index, so they stay free of the heavier study modules.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from tslab.core.paths import require_path

if TYPE_CHECKING:
    from tslab.study.definition import Study
    from tslab.study.machine import MachineProfile

__all__ = ['instance_directory', 'log_file_of', 'next_job_file', 'study_folder', 'usage_file']


def study_folder(*, machine: 'MachineProfile', study: 'Study') -> Path:
    """The hidden folder of the study: ``<output_root>/<study>/.study``."""
    return machine.output_root / study.name / '.study'


def log_file_of(*, folder: Path, job_name: str, index: int) -> Path:
    """The log file of one task: ``<folder>/logs/<job name>_<index>.out``.

    The index is numbered per job name, so the pair names one task of the study.
    """
    return folder / 'logs' / f'{job_name}_{index}.out'


def next_job_file(*, study_folder: Path) -> int:
    """The next job file number: one past the largest on disk, else 1."""
    numbers = [
        number
        for suffix in ('.sbatch', '.tasks')
        for path in (study_folder / 'jobs').glob(f'*{suffix}')
        if (number := _job_file_number(name=path.name, suffix=suffix)) is not None
    ]
    return max([0, *numbers]) + 1


def _job_file_number(*, name: str, suffix: str) -> int | None:
    """The number of a job file name, or ``None`` when the name holds no number."""
    try:
        return int(name.removesuffix(suffix))
    except ValueError:
        return None


def instance_directory(*, run_directory: Path, instance_id: str) -> Path:
    """The folder of one instance: ``stage[a/b]`` maps to ``stage/a/b``.

    Args:
        run_directory: The folder of the run.
        instance_id: The instance name from ``status.json``.

    Returns:
        The folder that holds the instance artifacts.
    """
    require_path(value=run_directory, parameter='run_directory')
    stage, _, name = instance_id.partition('[')
    return run_directory.joinpath(stage, *name.rstrip(']').split('/'))


def usage_file(*, log_file: Path) -> Path:
    """The usage file written next to the log of one local task.

    Args:
        log_file: The log file of the task.
    """
    return log_file.with_name(log_file.name + '.usage.json')
