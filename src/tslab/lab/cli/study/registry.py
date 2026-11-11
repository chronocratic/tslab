"""The table of ``tslab study`` subcommands."""

from types import MappingProxyType

from tslab.lab.cli.study.cancel import Cancel
from tslab.lab.cli.study.chain_exit import ChainExit
from tslab.lab.cli.study.failures import Failures, Usage
from tslab.lab.cli.study.list import List
from tslab.lab.cli.study.logs import Logs
from tslab.lab.cli.study.resubmit import Resubmit, RunOne
from tslab.lab.cli.study.results import Results
from tslab.lab.cli.study.slurm import Slurm
from tslab.lab.cli.study.status import Show, Status, Wait
from tslab.lab.cli.study.submit import Submit
from tslab.lab.study.command_names import CHAIN_EXIT_COMMAND

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

_INTERNAL_STUDY_COMMANDS = MappingProxyType({CHAIN_EXIT_COMMAND: ChainExit})
