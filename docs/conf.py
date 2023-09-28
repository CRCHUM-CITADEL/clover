# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Clover"
copyright = "2023, IVADO PRF3 Human Health"
author = "IVADO PRF3 Human Health"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx_autodoc_typehints",  # include signature hints
    "sphinx.ext.viewcode",  # view source code option
    "nbsphinx",  # include jupyter notebooks
    "nbsphinx_link",  # include files that are not in docs
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
        "color-problematic": "#00b377",
    },
    "dark_css_variables": {
        "color-brand-primary": "#926bff",
        "color-brand-content": "#926bff",
        "color-problematic": "#00b377",
    },
}
html_static_path = []
pygments_style = "sas"
pygments_dark_style = "monokai"


# Sorting parameter ('alphabetical' is default, 'groupwise' for member type and 'bysource' for source order)
autodoc_member_order = "bysource"

# Bug in smart quotes rendering for lists: ok solved with an escaped space
# smartquotes = False
