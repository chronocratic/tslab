"""Two-rank training entry point for the SLURM launcher integration test."""

import os
from pathlib import Path
import sys

from tests.ml.integration.lightning_toy import ToyData, ToyRegressor
from tests.support.integration_helpers import run_in
import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
from tslab.ml.stages.train import train


class _MultiRankRegressor(ToyRegressor):
    def fingerprint_settings(self) -> str:
        return 'multi-rank-regressor'


class _MultiRankData(ToyData):
    def fingerprint_settings(self) -> str:
        return 'multi-rank-data'


@tslab.lab.stage(name='after', output=tslab.lab.Json)
def after(*, run: tslab.lab.Run) -> tslab.lab.Json:
    del run
    return tslab.lab.Json(data={'rank': os.environ['SLURM_PROCID']})


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
