"""Sphinx configuration for the TSLab documentation."""

from tslab import __version__

project = 'TSLab'
author = 'The Chronocratic Developers'
copyright = '2026, The Chronocratic Developers'  # noqa: A001  # why: the name Sphinx reads
release = __version__
version = __version__

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'myst_parser']

# Google-style docstrings, as in the code.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_default_options = {'members': True, 'member-order': 'bysource'}
autodoc_typehints = 'description'

myst_enable_extensions = ['colon_fence', 'deflist']
# register heading slugs so cross-file [text](page.md#section) links resolve at build time
myst_heading_anchors = 3
source_suffix = {'.md': 'markdown'}
exclude_patterns = ['_build', 'superpowers']

html_theme = 'pydata_sphinx_theme'
html_title = 'TSLab'
html_theme_options = {
    'github_url': 'https://github.com/chronocratic/tslab',
    'navigation_depth': 2,
    'show_toc_level': 2,
}
html_static_path = ['_static']
