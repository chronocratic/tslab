"""The experiment script: one run that calls your stages.

Run ``python -m {{package_name}}.experiment --help`` for every setting.
"""

from dataclasses import dataclass, field
from pathlib import Path

import tslab.lab

from {{package_name}}.stages import measure


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """One run. Every field is a command-line flag.

    Attributes:
        scale: A setting passed to the stage.
        run: Resume and rerun options.
        run_path: Run folder under ``output_root``.
        output_root: Folder holding all runs.
        seed: Seeds the random number generators at the start of the run.
    """

    scale: float = 1.0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)
    run_path: Path = Path('experiment')
    output_root: Path = Path('runs')
    seed: int = 0


def main(*, config: ExperimentConfig) -> None:
    """Run the stages of the experiment in one run folder."""
    with tslab.lab.Run(
        config=config, root=config.output_root, relative_path=config.run_path, options=config.run
    ) as run:
        measure(run=run, scale=config.scale)


if __name__ == '__main__':
    main(config=tslab.lab.parse_config(config_class=ExperimentConfig))
