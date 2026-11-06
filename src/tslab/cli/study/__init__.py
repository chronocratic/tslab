"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses in the submodules hold only the flags of one command;
the behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

# why: the package door preserves command and helper imports while implementations live in siblings
from tslab.cli.study.cancel import Cancel
from tslab.cli.study.chain_exit import ChainExit as _ChainExit
from tslab.cli.study.failures import Failures, Usage
from tslab.cli.study.list import List
from tslab.cli.study.logs import Logs
from tslab.cli.study.resubmit import Resubmit, RunOne
from tslab.cli.study.results import Results
from tslab.cli.study.shared import (
    _context_of as _context_of,
    _detach as _detach,
    _log_next_step as _log_next_step,
    _missing_study_attribute_message as _missing_study_attribute_message,
    _missing_study_message as _missing_study_message,
    _selection_args as _selection_args,
    _selection_options as _selection_options,
    _states_of as _states_of,
    _status_hints as _status_hints,
    _study_lookup_error as _study_lookup_error,
    _usage_errors as _usage_errors,
    _warn_no_runs as _warn_no_runs,
    take_requested_code_snapshot as take_requested_code_snapshot,
)
from tslab.cli.study.slurm import Slurm
from tslab.cli.study.status import Show, Status, Wait
from tslab.cli.study.submit import Submit

__all__ = [
    'STUDY_COMMANDS',
    'Cancel',
    'Failures',
    'List',
    'Logs',
    'Resubmit',
    'Results',
    'RunOne',
    'Show',
    'Slurm',
    'Status',
    'Submit',
    'Usage',
    'Wait',
]

STUDY_COMMANDS = {
    'submit': Submit,
    'status': Status,
    'resubmit': Resubmit,
    'results': Results,
    'show': Show,
    'list': List,
    'wait': Wait,
    'failures': Failures,
    'logs': Logs,
    'cancel': Cancel,
    'usage': Usage,
    'slurm': Slurm,
    'run': RunOne,
}

_INTERNAL_STUDY_COMMANDS = {'chain-exit': _ChainExit}
