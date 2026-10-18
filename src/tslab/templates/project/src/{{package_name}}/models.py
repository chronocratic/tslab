"""The model choice shared by the scripts and the study."""

from tslab.adapters.chronocratic.models import chronocratic_model_choice

from {{package_name}}.augmentations import JitterThenScaleRecipe

Models = chronocratic_model_choice(augmentations=(JitterThenScaleRecipe,))
