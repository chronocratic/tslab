"""Two-rank training entry point for the SLURM launcher integration test."""

import os
from pathlib import Path
import sys

from integration_helpers import run_in
from lightning_toy import ToyData, ToyRegressor

import tslab
from tslab.lightning import LightningTrainer, LightningTrainerConfig
from tslab.stages import train


class _MultiRankRegressor(ToyRegressor):
    def fingerprint_settings(self) -> str:
        return 'multi-rank-regressor'


class _MultiRankData(ToyData):
    def fingerprint_settings(self) -> str:
        return 'multi-rank-data'


@tslab.stage(name='after', output=tslab.Json)
def after(*, run: tslab.Run) -> tslab.Json:
    del run
    return tslab.Json(data={'rank': os.environ['SLURM_PROCID']})


def main(*, run_root: Path) -> None:
    config = LightningTrainerConfig(
        accelerator='cpu', devices=2, max_epochs=2, extra_trainer_kwargs={'strategy': 'ddp'}
    )
    with run_in(root=run_root) as run:
        train(
            run=run,
            trainer=LightningTrainer(config=config),
            model=_MultiRankRegressor(),
            data=_MultiRankData(),
        )
        after(run=run)


if __name__ == '__main__':
    main(run_root=Path(sys.argv[1]))
