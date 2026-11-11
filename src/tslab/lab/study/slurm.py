"""Run one catalogue SLURM command for the jobs of a study, from a notebook."""

from collections.abc import Mapping
from pathlib import Path

from tslab.lab.jobs.slurm import run_slurm_command, slurm_command
from tslab.lab.study.definition.definition import Study
from tslab.lab.study.launchers.slurm.slurm_file import command_values
from tslab.lab.study.machine.load import load_machine
from tslab.lab.study.paths import study_folder

__all__ = ['slurm_output']


def slurm_output(*, study: Study, machine: Path, name: str, environment: Mapping[str, str]) -> str:
    """The output of one catalogue command, such as ``squeue-full-name``, for one study.

    Args:
        study: The study whose recorded SLURM jobs the command is about.
        machine: The machine profile file the study was submitted with.
        name: The catalogue name, ``<tool>-<variant>``.
        environment: The process environment used to read the machine profile.

    Returns:
        What the command printed on stdout.

    Raises:
        ValueError: If the name is unknown or the study has no recorded SLURM jobs.
        RuntimeError: If the command exits non-zero; the message is its stderr.
    """
    try:
        command = slurm_command(name=name)
    except KeyError as error:
        raise ValueError(error.args[0]) from error
    profile = load_machine(path=machine, environment=environment)
    values = command_values(study=study, study_folder=study_folder(machine=profile, study=study))
    if not values.job_ids:
        message = f'no SLURM jobs recorded for study {study.name!r}; submit it first'
        raise ValueError(message)
    output = run_slurm_command(command=command, values=values)
    if output.returncode != 0:
        message = output.stderr.strip() or f'{name} exited with {output.returncode}'
        raise RuntimeError(message)
    return output.stdout
