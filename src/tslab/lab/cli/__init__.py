"""The ``tslab`` command-line tool.

Experiments are never run through this tool: users run their own script
and the tool only offers helpers around runs. ``main`` is the entry point of the lab
alone: ``tslab new`` offers only the ``lab`` template.
"""

from tslab.lab.cli.entrypoint import main

__all__ = ['main']
