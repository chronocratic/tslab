"""The ``tslab study`` commands: submit, status, resubmit and results.

Each command is a plain function over a ``Context`` (see ``tslab.lab.study.context``).
The CLI dataclasses in ``tslab.lab.cli.study`` parse the arguments and call these
functions, so the behavior is testable without the command line.
"""
