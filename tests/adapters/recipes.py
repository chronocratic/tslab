"""Augmentation recipes for the adapter tests."""

from dataclasses import dataclass

from chronocratic.models.augmentation import (
    AlignedPair,
    AugmentationProducer,
    Jitter,
    JitterParameters,
    RolePairProducer,
    Scaling,
    ScalingParameters,
    ViewPair,
)
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer

RECIPE_CALLS: list[object] = []


@dataclass(frozen=True, kw_only=True)
class JitterThenScaleRecipe:
    """Jitter one view, scale the other."""

    sigma: float = 0.8
    ratio: float = 1.1

    def __call__(self) -> AugmentationProducer[ViewPair]:
        RECIPE_CALLS.append(self)
        return RolePairProducer(
            first=Jitter(params=JitterParameters(sigma=self.sigma)),
            second=Scaling(params=ScalingParameters(sigma=self.ratio)),
        )


@dataclass(frozen=True, kw_only=True)
class CropShiftRecipe:
    """TS2Vec's crop and shift, as a recipe."""

    def __call__(self) -> AugmentationProducer[AlignedPair]:
        return CropShiftProducer()
