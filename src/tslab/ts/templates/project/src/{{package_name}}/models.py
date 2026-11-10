"""The model choice shared by the scripts and the study."""

from typing import Any, TYPE_CHECKING

from tslab.ts.adapters.chronocratic.models import chronocratic_model_choice

from {{package_name}}.augmentations import JitterThenScaleRecipe

# The choice is built at run time; type checkers need a plain alias to accept it as an annotation.
if TYPE_CHECKING:
    type Models = Any
else:
    Models = chronocratic_model_choice(augmentations=(JitterThenScaleRecipe,))
