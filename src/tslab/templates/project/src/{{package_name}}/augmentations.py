"""Your augmentation recipes.

A recipe is a frozen dataclass of settings whose ``__call__`` builds a
chronocratic augmentation producer. Its return annotation names the view type,
and the recipe is offered only on models that take that view type:

- ``AugmentationProducer[ViewPair]``: TSTCC, MHCCL, SimCLR, CoST;
- ``AugmentationProducer[AlignedPair]``: TS2Vec;
- ``AugmentationProducer[SingleView]``: AutoTCL.

Pass recipes to ``chronocratic_model_choice(augmentations=(...))``; then
``model.augmentation:jitter-then-scale --model.augmentation.sigma 0.5`` selects
and sets one, and ``model.augmentation:default`` keeps the model's own. The
recipe's settings are part of the training fingerprint.
"""

from dataclasses import dataclass

from chronocratic.models.augmentation import (
    AugmentationProducer,
    Jitter,
    JitterParameters,
    RolePairProducer,
    Scaling,
    ScalingParameters,
    ViewPair,
)


@dataclass(frozen=True)
class JitterThenScaleRecipe:
    """Jitter one view and scale the other.

    Attributes:
        sigma: Standard deviation of the jitter noise.
        ratio: Standard deviation of the scaling factor.
    """

    sigma: float = 0.8
    ratio: float = 1.1

    def __call__(self) -> AugmentationProducer[ViewPair]:
        """Build the producer."""
        return RolePairProducer(
            first=Jitter(params=JitterParameters(sigma=self.sigma)),
            second=Scaling(params=ScalingParameters(sigma=self.ratio)),
        )
