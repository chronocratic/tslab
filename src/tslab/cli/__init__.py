"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs.
"""

from tslab.cli.entrypoint import main

__all__ = ['main']
