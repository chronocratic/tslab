"""``LightningDataSource``: use a ``LightningDataModule`` wherever TSLab expects a data source."""

from collections.abc import Iterable
from typing import cast

from lightning import LightningDataModule
from lightning.pytorch.utilities.model_helpers import is_overridden

from tslab.interfaces.data import Batch

__all__ = ['LightningDataSource']


class LightningDataSource[I]:
    """Adapts a ``LightningDataModule`` to ``DataSource[I]`` (Adapter pattern).

    ``prepare_data()`` runs once and ``setup(stage)`` once per stage, the
    first time a loader of that stage is requested. ``LightningTrainer`` hands
    the datamodule itself to Lightning, so its hooks run as usual.
    """

    def __init__(self, *, datamodule: LightningDataModule, info: I) -> None:
        """Wrap ``datamodule``.

        Args:
            datamodule: The datamodule providing the loaders.
            info: Facts about the dataset (for example ``tslab.tasks.TSDataInfo``).
        """
        self._datamodule = datamodule
        self._info = info
        self._prepared = False
        self._stages_set_up: set[str] = set()

    @property
    def info(self) -> I:
        """Return facts about the dataset."""
        return self._info

    @property
    def datamodule(self) -> LightningDataModule:
        """Return the wrapped datamodule."""
        return self._datamodule

    @property
    def has_validation(self) -> bool:
        """Return whether the datamodule defines a validation loader (without setting it up)."""
        return is_overridden(method_name='val_dataloader', instance=self._datamodule)

    def train_loader(self) -> Iterable[Batch]:
        """Return the training loader."""
        self._set_up(stage='fit')
        # Lightning types loaders as Any; a datamodule's loaders yield batches.
        return cast('Iterable[Batch]', self._datamodule.train_dataloader())

    def val_loader(self) -> Iterable[Batch] | None:
        """Return the validation loader, or ``None`` if the datamodule has none."""
        if not self.has_validation:
            return None
        self._set_up(stage='fit')
        return cast('Iterable[Batch]', self._datamodule.val_dataloader())

    def test_loader(self) -> Iterable[Batch]:
        """Return the test loader."""
        self._set_up(stage='test')
        return cast('Iterable[Batch]', self._datamodule.test_dataloader())

    def _set_up(self, *, stage: str) -> None:
        if not self._prepared:
            self._datamodule.prepare_data()
            self._prepared = True
        if stage not in self._stages_set_up:
            self._datamodule.setup(stage=stage)
            self._stages_set_up.add(stage)
