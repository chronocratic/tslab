"""Reading local task states from the pid file and job records."""

from tslab.core.meta import hostname
from tslab.study.definition import Study
from tslab.study.launchers.local.pid_file import local_job_states
from tslab.study.machine import MachineProfile
from tslab.study.plan import study_folder
from tslab.study.records import JobRecord, latest_tasks

__all__ = ['_is_local_job_id', '_local_states']


def _is_local_job_id(*, job_id: str) -> bool:
    """Whether a job id has the process-id and file-number shape."""
    prefix, first_separator, remainder = job_id.partition('-')
    process_id, second_separator, job_file = remainder.partition('-')
    return bool(
        first_separator
        and second_separator
        and prefix.isalpha()
        and process_id.isdecimal()
        and job_file.isdecimal()
    )


def _local_states(
    *, machine: MachineProfile, study: Study, jobs: list[JobRecord]
) -> dict[str, str]:
    live = local_job_states(
        study_folder=study_folder(machine=machine, study=study), host=hostname()
    )
    # why: a local task outside a live launch is over, whatever its folder says
    ended = {
        f'{job.job_id}_{task.index}': 'ended'
        for job, task in latest_tasks(jobs=jobs).values()
        if _is_local_job_id(job_id=job.job_id)
    }
    return {**ended, **live}
