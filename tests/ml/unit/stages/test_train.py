from pathlib import Path

from tests.ts.unit.stage_fakes import FakeTrainer, LinearEncoder, run_in, ToySplits
import tslab.lab
import tslab.ml


def test_train_writes_its_checkpoint_into_the_instance_folder(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        checkpoint = tslab.ml.stages.train(
            run=run, trainer=FakeTrainer(), model=LinearEncoder(), data=ToySplits()
        )
    folder = tmp_path / 'r' / 'train' / 'default'
    assert checkpoint.directory == folder
    assert sorted(path.name for path in folder.iterdir()) == [
        'checkpoint.json',
        'last.ckpt',
        'meta.json',
    ]


def test_finished_training_is_loaded_not_repeated(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        tslab.ml.stages.train(
            run=run, trainer=FakeTrainer(), model=LinearEncoder(), data=ToySplits()
        )
    trainer = FakeTrainer()
    with run_in(root=tmp_path) as run:
        checkpoint = tslab.ml.stages.train(
            run=run, trainer=trainer, model=LinearEncoder(), data=ToySplits()
        )
    assert trainer.fit_calls == 0
    assert checkpoint.best == 'last.ckpt'


def test_failed_training_resumes_in_the_same_folder(tmp_path: Path) -> None:
    with run_in(root=tmp_path, fail_fast=False) as run:
        failed = tslab.ml.stages.train(
            run=run, trainer=FakeTrainer(fail=True), model=LinearEncoder(), data=ToySplits()
        )
    assert not tslab.lab.is_available(artifact=failed)
    assert (tmp_path / 'r' / 'train' / 'default' / '.incomplete').is_file()
    trainer = FakeTrainer()
    with run_in(root=tmp_path) as run:
        tslab.ml.stages.train(run=run, trainer=trainer, model=LinearEncoder(), data=ToySplits())
    assert trainer.resumed
    assert not (tmp_path / 'r' / 'train' / 'default' / '.incomplete').exists()
